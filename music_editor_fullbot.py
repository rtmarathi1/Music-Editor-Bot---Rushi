# music_editor_fullbot.py

import os
import json
import asyncio
import subprocess
from pathlib import Path
from tempfile import mkstemp
from typing import Optional, Tuple
import aiofiles

from telegram import Update, InputFile
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
SETTINGS_FILE = "bot_settings.json"
STATS_FILE = "bot_stats.json"

MAX_FILE_SIZE_BYTES = 30 * 1024 * 1024
CONCURRENT_JOBS = 2
JOB_TIMEOUT_SECONDS = 60
job_semaphore = asyncio.Semaphore(CONCURRENT_JOBS)


# ---------------- JSON HELPERS ----------------
def load_json(path, default):
    try:
        with open(path, "r", encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


settings = load_json(SETTINGS_FILE, {"chats": {}, "channels": {}})
stats = load_json(STATS_FILE, {"total_processed": 0, "by_user": {}})


def ensure_chat_settings(chat_id):
    ks = str(chat_id)
    if ks not in settings["chats"]:
        settings["chats"][ks] = {
            "artist": None,
            "picture_file_id": None,
            "last_edit_params": {}
        }
    return settings["chats"][ks]


# ---------------- UTILITIES ----------------
async def save_file_from_telegram(file_obj, dest_path: str):
    file = await file_obj.get_file()
    await file.download_to_drive(dest_path)
    return dest_path


def ffmpeg_run(cmd: list, timeout: int = JOB_TIMEOUT_SECONDS) -> Tuple[int, bytes, bytes]:
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        proc.kill()
        return -1, b"", b"timeout"
    except Exception as e:
        return -2, b"", str(e).encode()


def build_atempo_filters(factor: float) -> str:
    if factor <= 0:
        factor = 1.0
    if 0.5 <= factor <= 2.0:
        return f"atempo={factor}"

    filters = []
    remaining = factor
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5

    filters.append(f"atempo={remaining}")
    return ",".join(filters)


def build_pitch_filter(semitones: float) -> Optional[str]:
    if semitones == 0:
        return None
    factor = 2 ** (semitones / 12.0)
    return f"asetrate=44100*{factor},aresample=44100"


def format_time_arg(value: str) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    try:
        float(value)
        return value
    except:
        return value


# ---------------- BASIC COMMANDS ----------------
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Music Editor Bot ready. Use /help for commands.")


async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    txt = (
        "Commands:\n"
        "/setartist <name>\n"
        "/setpic\n"
        "/connect_channel\n\n"
        "Editing:\n"
        "/speed <factor>\n"
        "/pitch <semitones>\n"
        "/trim <start> <end>\n"
        "/convert <format>\n\n"
        "/preview\n"
        "/apply\n"
        "/post_to_channel\n\n"
        "Stats:\n"
        "/stats"
    )
    await update.message.reply_text(txt)


async def set_artist(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Usage: /setartist <artist name>")
        return
    cs = ensure_chat_settings(update.effective_chat.id)
    cs["artist"] = " ".join(ctx.args).strip()
    save_json(SETTINGS_FILE, settings)
    await update.message.reply_text(f"Artist set to: {cs['artist']}")


async def set_pic_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data['awaiting_pic_for_chat'] = update.effective_chat.id
    await update.message.reply_text("Send the photo you want as album art.")


async def photo_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    awaiting_chat = ctx.user_data.get('awaiting_pic_for_chat')
    if awaiting_chat and awaiting_chat == update.effective_chat.id:
        photo = update.message.photo[-1]
        file_id = photo.file_id
        cs = ensure_chat_settings(awaiting_chat)
        cs["picture_file_id"] = file_id
        save_json(SETTINGS_FILE, settings)
        ctx.user_data.pop('awaiting_pic_for_chat', None)
        await update.message.reply_text("Album art saved.")
        return

    awaiting_channel = ctx.user_data.get('awaiting_channel_connect')
    if awaiting_channel and update.message.forward_from_chat and update.message.forward_from_chat.type == "channel":
        channel = update.message.forward_from_chat
        settings["channels"][str(channel.id)] = {"title": channel.title, "username": channel.username}
        save_json(SETTINGS_FILE, settings)
        ctx.user_data.pop('awaiting_channel_connect', None)
        await update.message.reply_text(f"Channel registered: {channel.title}")
        return


async def connect_channel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if ctx.args:
        alias = ctx.args[0].strip()
        try:
            chat = await ctx.bot.get_chat(alias)
        except:
            await update.message.reply_text("Cannot find channel.")
            return
        settings["channels"][str(chat.id)] = {"title": chat.title, "username": chat.username}
        save_json(SETTINGS_FILE, settings)
        await update.message.reply_text(f"Channel connected: {chat.title}")
        return

    ctx.user_data['awaiting_channel_connect'] = True
    await update.message.reply_text("Forward any channel post here to connect it.")


async def stats_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    t = stats.get("total_processed", 0)
    await update.message.reply_text(f"Total processed: {t}")


# ---------------- PARAMETER COMMANDS ----------------
def set_last_params_for_chat(chat_id, params: dict):
    cs = ensure_chat_settings(chat_id)
    cs["last_edit_params"] = params
    save_json(SETTINGS_FILE, settings)


def get_last_params_for_chat(chat_id) -> dict:
    cs = ensure_chat_settings(chat_id)
    return cs.get("last_edit_params", {})


async def speed_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Usage: /speed <factor>")
        return
    try:
        f = float(ctx.args[0])
    except:
        await update.message.reply_text("Invalid number.")
        return
    params = get_last_params_for_chat(update.effective_chat.id)
    params["speed"] = f
    set_last_params_for_chat(update.effective_chat.id, params)
    await update.message.reply_text(f"Speed set: {f}")


async def pitch_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Usage: /pitch <semitones>")
        return
    try:
        s = float(ctx.args[0])
    except:
        await update.message.reply_text("Invalid number.")
        return
    params = get_last_params_for_chat(update.effective_chat.id)
    params["pitch"] = s
    set_last_params_for_chat(update.effective_chat.id, params)
    await update.message.reply_text(f"Pitch set: {s}")


async def trim_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) < 2:
        await update.message.reply_text("Usage: /trim <start> <end>")
        return
    start = format_time_arg(ctx.args[0])
    end = format_time_arg(ctx.args[1])
    params = get_last_params_for_chat(update.effective_chat.id)
    params["trim_start"] = start
    params["trim_end"] = end
    set_last_params_for_chat(update.effective_chat.id, params)
    await update.message.reply_text(f"Trim: {start} → {end}")


async def convert_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Usage: /convert <format>")
        return
    fmt = ctx.args[0].lower().strip()
    params = get_last_params_for_chat(update.effective_chat.id)
    params["convert_to"] = fmt
    set_last_params_for_chat(update.effective_chat.id, params)
    await update.message.reply_text(f"Format set: {fmt}")


# ---------------- AUDIO PROCESSING ----------------
async def process_audio_file(in_path: str, out_path: str, params: dict) -> Tuple[bool, str]:
    filters = []

    if "speed" in params:
        filters.append(build_atempo_filters(float(params["speed"])))

    if "pitch" in params:
        pf = build_pitch_filter(float(params["pitch"]))
        if pf:
            filters.append(pf)

    filter_str = ",".join(filters) if filters else None

    cmd = ["ffmpeg", "-y"]
    if params.get("trim_start"):
        cmd += ["-ss", str(params["trim_start"])]
    cmd += ["-i", in_path]

    if params.get("trim_end"):
        cmd += ["-to", str(params["trim_end"])]

    if filter_str:
        cmd += ["-filter:a", filter_str]

    outfmt = params.get("convert_to", "mp3")

    if outfmt == "wav":
        cmd += ["-vn", "-ac", "2", "-ar", "44100", out_path]
    elif outfmt == "ogg":
        cmd += ["-vn", "-acodec", "libvorbis", "-ar", "44100", out_path]
    elif outfmt in ["m4a", "aac"]:
        cmd += ["-vn", "-c:a", "aac", "-b:a", "192k", out_path]
    else:
        cmd += ["-vn", "-ac", "2", "-ar", "44100", "-b:a", "192k", out_path]

    loop = asyncio.get_event_loop()
    rc, out, err = await loop.run_in_executor(None, ffmpeg_run, cmd)

    return (rc == 0, err.decode(errors="ignore"))


def file_size_ok(path: str) -> bool:
    try:
        return Path(path).stat().st_size <= MAX_FILE_SIZE_BYTES
    except:
        return False


async def handle_audio_upload(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    file_obj = msg.voice or msg.audio or (msg.document if msg.document and msg.document.mime_type and msg.document.mime_type.startswith("audio/") else None)

    if not file_obj:
        await msg.reply_text("Send an audio file.")
        return

    fd, in_path = mkstemp()
    os.close(fd)

    try:
        await save_file_from_telegram(file_obj, in_path)
    except:
        await msg.reply_text("Download failed.")
        return

    if not file_size_ok(in_path):
        await msg.reply_text("File too large.")
        return

    ctx.user_data["last_uploaded_audio"] = in_path
    await msg.reply_text("Audio received. Use /preview, /apply, or /post_to_channel.")


async def preview_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat.id
    in_path = ctx.user_data.get("last_uploaded_audio")

    if not in_path or not Path(in_path).exists():
        await update.message.reply_text("Send audio first.")
        return

    params = get_last_params_for_chat(chat)
    params_preview = params.copy()
    params_preview["trim_start"] = 0
    params_preview["trim_end"] = 10

    fd, out_path = mkstemp(suffix=".mp3")
    os.close(fd)

    await update.message.reply_text("Generating preview...")

    async with job_semaphore:
        ok, err = await process_audio_file(in_path, out_path, params_preview)

    if not ok:
        await update.message.reply_text(f"Preview failed: {err}")
        return

    cs = ensure_chat_settings(chat)
    performer = cs.get("artist")
    thumb = cs.get("picture_file_id")

    with open(out_path, "rb") as f:
        await ctx.bot.send_audio(
            chat_id=chat,
            audio=f,
            performer=performer,
            thumb=thumb,
            filename="preview.mp3"
        )


async def apply_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat.id
    in_path = ctx.user_data.get("last_uploaded_audio")

    if not in_path or not Path(in_path).exists():
        await update.message.reply_text("Send audio first.")
        return

    params = get_last_params_for_chat(chat)

    fmt = params.get("convert_to", "mp3")
    fd, out_path = mkstemp(suffix=f".{fmt}")
    os.close(fd)

    await update.message.reply_text("Processing full file...")

    async with job_semaphore:
        ok, err = await process_audio_file(in_path, out_path, params)

    if not ok:
        await update.message.reply_text(f"Error: {err}")
        return

    stats["total_processed"] += 1
    save_json(STATS_FILE, stats)

    cs = ensure_chat_settings(chat)
    performer = cs.get("artist")
    thumb = cs.get("picture_file_id")

    with open(out_path, "rb") as f:
        await ctx.bot.send_audio(
            chat_id=chat,
            audio=f,
            performer=performer,
            thumb=thumb,
            filename=f"edited.{fmt}"
        )


async def post_to_channel_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat.id
    in_path = ctx.user_data.get("last_uploaded_audio")

    if not in_path or not Path(in_path).exists():
        await update.message.reply_text("Send audio first.")
        return

    if not settings["channels"]:
        await update.message.reply_text("No channels connected.")
        return

    channel_id = list(settings["channels"].keys())[0]
    params = get_last_params_for_chat(chat)
    fmt = params.get("convert_to", "mp3")

    fd, out_path = mkstemp(suffix=f".{fmt}")
    os.close(fd)

    await update.message.reply_text("Posting to channel...")

    async with job_semaphore:
        ok, err = await process_audio_file(in_path, out_path, params)

    if not ok:
        await update.message.reply_text(f"Failed: {err}")
        return

    cs = ensure_chat_settings(chat)
    performer = cs.get("artist")
    thumb = cs.get("picture_file_id")

    with open(out_path, "rb") as f:
        await ctx.bot.send_audio(
            chat_id=int(channel_id),
            audio=f,
            performer=performer,
            thumb=thumb,
            filename=f"channel_post.{fmt}"
        )

    await update.message.reply_text("Posted.")


# ---------------- MAIN ----------------
def main():
    if not BOT_TOKEN:
        print("ERROR: BOT_TOKEN not set.")
        return

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("setartist", set_artist))
    app.add_handler(CommandHandler("setpic", set_pic_command))
    app.add_handler(CommandHandler("connect_channel", connect_channel))
    app.add_handler(CommandHandler("stats", stats_cmd))

    app.add_handler(CommandHandler("speed", speed_cmd))
    app.add_handler(CommandHandler("pitch", pitch_cmd))
    app.add_handler(CommandHandler("trim", trim_cmd))
    app.add_handler(CommandHandler("convert", convert_cmd))

    app.add_handler(CommandHandler("preview", preview_cmd))
    app.add_handler(CommandHandler("apply", apply_cmd))
    app.add_handler(CommandHandler("post_to_channel", post_to_channel_cmd))

    app.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, photo_handler))

    app.add_handler(MessageHandler(
        filters.VOICE | filters.AUDIO |
        (filters.Document & filters.Document.file_mime_type("audio/.*")),
        handle_audio_upload
    ))

    print("Starting Music Editor Full Bot...")
    app.run_polling()


if __name__ == "__main__":
    main()

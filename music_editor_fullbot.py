# music_editor_fullbot.py
# Requirements:
#   pip install python-telegram-bot==20.5 aiofiles
#   ffmpeg available on PATH
#
# Usage: set BOT_TOKEN env var and run: python music_editor_fullbot.py

import os
import json
import asyncio
import shlex
import subprocess
from pathlib import Path
from tempfile import mkstemp
from typing import Optional, Tuple
import aiofiles

from telegram import Update, InputFile
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)

BOT_TOKEN = os.environ.get("BOT_TOKEN", "YOUR_TOKEN_HERE")
SETTINGS_FILE = "bot_settings.json"
STATS_FILE = "bot_stats.json"

# limits (tune for your host)
MAX_FILE_SIZE_BYTES = 30 * 1024 * 1024  # 30 MB
CONCURRENT_JOBS = 2  # simple concurrency limit (semaphore)
JOB_TIMEOUT_SECONDS = 60  # per ffmpeg run timeout

# simple semaphore to prevent overload
job_semaphore = asyncio.Semaphore(CONCURRENT_JOBS)

# ========== JSON storage helpers ==========
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
            "last_edit_params": {}  # hold current edit params for preview/apply
        }
    return settings["chats"][ks]

# ========== utilities ==========
async def save_file_from_telegram(file_obj, dest_path: str):
    """Download Telegram File to dest_path using bot's get_file()."""
    file = await file_obj.get_file()
    await file.download_to_drive(dest_path)
    return dest_path

def ffmpeg_run(cmd: list, timeout: int = JOB_TIMEOUT_SECONDS) -> Tuple[int, bytes, bytes]:
    """Run ffmpeg command synchronously (called inside thread via asyncio)."""
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
    """Return atempo filter string (chains if needed). Limitations: atempo only 0.5-2.0 per filter."""
    if factor <= 0:
        factor = 1.0
    if 0.5 <= factor <= 2.0:
        return f"atempo={factor}"
    filters = []
    remaining = factor
    # decompose multiplicatively into factors between 0.5 and 2.0
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining}")
    return ",".join(filters)

def build_pitch_filter(semitones: float) -> Optional[str]:
    """Simple pitch shift via asetrate trick (changes duration). Good for small shifts."""
    if semitones == 0:
        return None
    # pitch factor: 2^(semitones/12)
    factor = 2 ** (semitones / 12.0)
    # change sample rate then resample back to 44100
    return f"asetrate=44100*{factor},aresample=44100"

def format_time_arg(value: str) -> Optional[str]:
    """Accept seconds (float) or HH:MM:SS and return string for ffmpeg."""
    if not value:
        return None
    value = value.strip()
    # if numeric
    try:
        float(value)
        return value
    except:
        # assume it's HH:MM:SS already
        return value

# ========== command handlers ==========
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Music Editor Bot ready. Use /help for commands.")

async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    txt = (
        "Commands:\n"
        "/setartist <name> — set artist metadata\n"
        "/setpic — send this, then send a photo to save as album art\n"
        "/connect_channel — run then forward a message from your channel OR /connect_channel @channelusername\n\n"
        "Editing commands (use before sending audio):\n"
        "/speed <factor> — e.g. /speed 1.25 or /speed 0.8\n"
        "/pitch <semitones> — e.g. /pitch 2 or /pitch -3\n"
        "/trim <start_seconds> <end_seconds> — e.g. /trim 5 20\n"
        "/convert <format> — e.g. /convert mp3 or /convert wav\n\n"
        "When you have set params, send audio/voice and use:\n"
        "/preview — get a short preview (10s) showing edits\n"
        "/apply — process and return full edited file\n"
        "/post_to_channel — process & post to connected channel (admin + bot must be admin)\n\n"
        "Also /stats and /help\n"
    )
    await update.message.reply_text(txt)

# set artist & pic & connect_channel & stats reused from previous module
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
    await update.message.reply_text("Send the photo you want to use as album art (reply with photo).")

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

    # channel-forward registration (if awaiting)
    awaiting_channel = ctx.user_data.get('awaiting_channel_connect')
    if awaiting_channel and update.message.forward_from_chat and update.message.forward_from_chat.type == "channel":
        channel = update.message.forward_from_chat
        settings["channels"][str(channel.id)] = {"title": channel.title, "username": channel.username}
        save_json(SETTINGS_FILE, settings)
        ctx.user_data.pop('awaiting_channel_connect', None)
        await update.message.reply_text(f"Channel registered: {channel.title} (id: {channel.id})")
        return

async def connect_channel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    app = ctx.application
    # allow /connect_channel @username
    if ctx.args:
        target = ctx.args[0].strip()
        try:
            target_chat = await app.bot.get_chat(target)
        except Exception:
            await update.message.reply_text("Couldn't find the channel. Try forwarding a message from the channel after running /connect_channel.")
            return
        # store
        settings["channels"][str(target_chat.id)] = {"title": target_chat.title, "username": target_chat.username}
        save_json(SETTINGS_FILE, settings)
        await update.message.reply_text(f"Channel connected: {target_chat.title} (id:{target_chat.id})")
        return
    ctx.user_data['awaiting_channel_connect'] = True
    await update.message.reply_text("Run this command then forward any message from the channel to register it. Bot must be an admin in the channel.")

async def stats_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    total = stats.get("total_processed", 0)
    by_user = stats.get("by_user", {})
    top = sorted(by_user.items(), key=lambda x: -x[1])[:10]
    lines = [f"Total processed audios: {total}", "Top chats:"]
    for uid, cnt in top:
        lines.append(f"- {uid}: {cnt}")
    await update.message.reply_text("\n".join(lines) if lines else "No stats yet.")

# ========== editing parameter commands ==========
def set_last_params_for_chat(chat_id, params: dict):
    cs = ensure_chat_settings(chat_id)
    cs["last_edit_params"] = params
    save_json(SETTINGS_FILE, settings)

def get_last_params_for_chat(chat_id) -> dict:
    cs = ensure_chat_settings(chat_id)
    return cs.get("last_edit_params", {})

async def speed_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Usage: /speed <factor> e.g. /speed 1.25")
        return
    try:
        f = float(ctx.args[0])
    except:
        await update.message.reply_text("Invalid number.")
        return
    params = get_last_params_for_chat(update.effective_chat.id)
    params["speed"] = f
    set_last_params_for_chat(update.effective_chat.id, params)
    await update.message.reply_text(f"Speed factor set to {f}")

async def pitch_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Usage: /pitch <semitones> e.g. /pitch -2")
        return
    try:
        s = float(ctx.args[0])
    except:
        await update.message.reply_text("Invalid semitone number.")
        return
    params = get_last_params_for_chat(update.effective_chat.id)
    params["pitch"] = s
    set_last_params_for_chat(update.effective_chat.id, params)
    await update.message.reply_text(f"Pitch shift set to {s} semitones")

async def trim_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) < 2:
        await update.message.reply_text("Usage: /trim <start_seconds> <end_seconds> e.g. /trim 5 20")
        return
    start = format_time_arg(ctx.args[0])
    end = format_time_arg(ctx.args[1])
    params = get_last_params_for_chat(update.effective_chat.id)
    params["trim_start"] = start
    params["trim_end"] = end
    set_last_params_for_chat(update.effective_chat.id, params)
    await update.message.reply_text(f"Trim set: {start} -> {end}")

async def convert_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) < 1:
        await update.message.reply_text("Usage: /convert <format> e.g. /convert mp3")
        return
    fmt = ctx.args[0].lower().strip()
    params = get_last_params_for_chat(update.effective_chat.id)
    params["convert_to"] = fmt
    set_last_params_for_chat(update.effective_chat.id, params)
    await update.message.reply_text(f"Target format set to: {fmt}")

# ========== core process function ==========
async def process_audio_file(in_path: str, out_path: str, params: dict) -> Tuple[bool, str]:
    """
    Builds ffmpeg command based on params and runs it.
    params may contain keys: speed, pitch, trim_start, trim_end, convert_to
    """
    filters = []
    # speed
    if "speed" in params:
        atempo = build_atempo_filters(float(params["speed"]))
        if atempo:
            filters.append(atempo)
    # pitch
    if "pitch" in params:
        pf = build_pitch_filter(float(params["pitch"]))
        if pf:
            filters.append(pf)
    filter_str = ",".join(filters) if filters else None

    # trim args
    cmd = ["ffmpeg", "-y"]
    if params.get("trim_start"):
        cmd += ["-ss", str(params["trim_start"])]
    cmd += ["-i", in_path]
    if params.get("trim_end"):
        # if -ss before -i then use -to after -i (duration or absolute)
        cmd += ["-to", str(params["trim_end"])]
    if filter_str:
        cmd += ["-filter:a", filter_str]
    # output options
    outfmt = params.get("convert_to", None)
    # choose container/codec
    if outfmt == "wav":
        cmd += ["-vn", "-ac", "2", "-ar", "44100", out_path]
    elif outfmt == "ogg":
        cmd += ["-vn", "-acodec", "libvorbis", "-ar", "44100", out_path]
    elif outfmt == "m4a" or outfmt == "aac":
        cmd += ["-vn", "-c:a", "aac", "-b:a", "192k", out_path]
    else:
        # default mp3
        cmd += ["-vn", "-ac", "2", "-ar", "44100", "-b:a", "192k", out_path]
    # run in thread to avoid blocking
    loop = asyncio.get_event_loop()
    rc, out, err = await loop.run_in_executor(None, ffmpeg_run, cmd)
    if rc == 0:
        return True, ""
    else:
        return False, err.decode(errors="ignore")

# ========== audio upload handlers and preview/apply ==========
def file_size_ok(path: str) -> bool:
    try:
        return Path(path).stat().st_size <= MAX_FILE_SIZE_BYTES
    except:
        return False

async def handle_audio_upload(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """When an audio/voice is sent, we save it and store a pointer in user_data so /preview or /apply can use it."""
    msg = update.message
    file_obj = msg.voice or msg.audio or (msg.document if msg.document and msg.document.mime_type and msg.document.mime_type.startswith("audio/") else None)
    if not file_obj:
        await msg.reply_text("Send an audio file or voice note.")
        return
    # download to temp
    fd_in, in_path = mkstemp(suffix=".ogg")
    os.close(fd_in)
    try:
        # file.get_file() and download
        await save_file_from_telegram(file_obj, in_path)
    except Exception as e:
        await msg.reply_text("Failed to download file.")
        return
    if not file_size_ok(in_path):
        await msg.reply_text("File too large. Limit is ~30 MB.")
        try:
            os.remove(in_path)
        except: pass
        return
    # save path in user_data for later preview/apply
    ctx.user_data["last_uploaded_audio"] = in_path
    await msg.reply_text("Audio received. Use /preview to get a short preview, or /apply to process the full file. Use /post_to_channel to publish to channel.")

async def preview_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    in_path = ctx.user_data.get("last_uploaded_audio")
    if not in_path or not Path(in_path).exists():
        await update.message.reply_text("No uploaded audio found. Send audio first.")
        return
    params = get_last_params_for_chat(chat_id).copy()
    # for preview, enforce a 10s clip: use -t 10
    params_preview = params.copy()
    # we'll use -t 10 by adding trim_start = 0 and trim_end = 10 if not present
    if not params_preview.get("trim_start"):
        params_preview["trim_start"] = 0
    params_preview["trim_end"] = 10
    out_fd, out_path = mkstemp(suffix=".mp3")
    os.close(out_fd)

    await update.message.reply_text("Processing preview...")

    # concurrency limit
    async with job_semaphore:
        ok, err = await process_audio_file(in_path, out_path, params_preview)
    if not ok:
        await update.message.reply_text(f"Preview failed: {err}")
        try:
            os.remove(out_path)
        except: pass
        return

    # send preview with metadata thumbnail/performer
    cs = ensure_chat_settings(str(chat_id))
    performer = cs.get("artist")
    thumb = cs.get("picture_file_id")
    try:
        with open(out_path, "rb") as f:
            kwargs = {}
            if performer:
                kwargs["performer"] = performer
            if thumb:
                await ctx.bot.send_audio(chat_id=chat_id, audio=InputFile(f, filename="preview.mp3"), thumb=thumb, **kwargs)
            else:
                await ctx.bot.send_audio(chat_id=chat_id, audio=InputFile(f, filename="preview.mp3"), **kwargs)
    except Exception as e:
        await update.message.reply_text("Failed to send preview; sending as document.")
        await ctx.bot.send_document(chat_id=chat_id, document=InputFile(out_path, filename="preview.mp3"))
    finally:
        try: os.remove(out_path)
        except: pass

async def apply_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    in_path = ctx.user_data.get("last_uploaded_audio")
    if not in_path or not Path(in_path).exists():
        await update.message.reply_text("No uploaded audio found. Send audio first.")
        return
    params = get_last_params_for_chat(chat_id) or {}
    # determine output extension
    fmt = params.get("convert_to", "mp3")
    suffix = f".{fmt if fmt else 'mp3'}"
    out_fd, out_path = mkstemp(suffix=suffix)
    os.close(out_fd)

    await update.message.reply_text("Processing full file... This may take a moment.")

    async with job_semaphore:
        ok, err = await process_audio_file(in_path, out_path, params)
    if not ok:
        await update.message.reply_text(f"Processing failed: {err}")
        try: os.remove(out_path)
        except: pass
        return

    # update stats
    stats["total_processed"] = stats.get("total_processed", 0) + 1
    uid = str(chat_id)
    stats.setdefault("by_user", {})
    stats["by_user"][uid] = stats["by_user"].get(uid, 0) + 1
    save_json(STATS_FILE, stats)

    # send final file with metadata
    cs = ensure_chat_settings(chat_id)
    performer = cs.get("artist")
    thumb = cs.get("picture_file_id")
    try:
        with open(out_path, "rb") as f:
            kwargs = {}
            if performer:
                kwargs["performer"] = performer
            # send as audio (so it becomes music with metadata) — if thumbnail present, use it
            if thumb:
                await ctx.bot.send_audio(chat_id=chat_id, audio=InputFile(f, filename=f"edited{suffix}"), thumb=thumb, **kwargs)
            else:
                await ctx.bot.send_audio(chat_id=chat_id, audio=InputFile(f, filename=f"edited{suffix}"), **kwargs)
    except Exception as e:
        # fallback to document
        await ctx.bot.send_document(chat_id=chat_id, document=InputFile(out_path, filename=f"edited{suffix}"))
    finally:
        try:
            os.remove(out_path)
        except:
            pass

async def post_to_channel_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Process and post to first connected channel for this bot (simple behavior)."""
    chat_id = update.effective_chat.id
    in_path = ctx.user_data.get("last_uploaded_audio")
    if not in_path or not Path(in_path).exists():
        await update.message.reply_text("No uploaded audio found. Send audio first.")
        return
    # find a channel registered (simple choose first)
    if not settings.get("channels"):
        await update.message.reply_text("No channels registered. Use /connect_channel first.")
        return
    # pick first channel id
    channel_id = next(iter(settings["channels"].keys()))
    params = get_last_params_for_chat(chat_id) or {}
    fmt = params.get("convert_to", "mp3")
    suffix = f".{fmt}"
    out_fd, out_path = mkstemp(suffix=suffix)
    os.close(out_fd)
    await update.message.reply_text("Processing and posting to connected channel...")

    async with job_semaphore:
        ok, err = await process_audio_file(in_path, out_path, params)
    if not ok:
        await update.message.reply_text(f"Processing failed: {err}")
        try: os.remove(out_path)
        except: pass
        return

    # send to channel
    cs = ensure_chat_settings(chat_id)
    performer = cs.get("artist")
    thumb = cs.get("picture_file_id")
    try:
        with open(out_path, "rb") as f:
            kwargs = {}
            if performer:
                kwargs["performer"] = performer
            await ctx.bot.send_audio(chat_id=int(channel_id), audio=InputFile(f, filename=f"channel_post{suffix}"), thumb=thumb, **kwargs)
        await update.message.reply_text(f"Posted to channel (id: {channel_id}).")
    except Exception as e:
        await update.message.reply_text(f"Failed to post to channel: {e}")
    finally:
        try: os.remove(out_path)
        except: pass

# cleanup helper to remove user's last uploaded file on exit or when replaced
async def cleanup_user_uploads_on_replace(ctx_user_data):
    # not implemented automatically; old files replaced when new audio uploaded by handler

# ========== wiring ==========
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # basic commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("setartist", set_artist))
    app.add_handler(CommandHandler("setpic", set_pic_command))
    app.add_handler(CommandHandler("connect_channel", connect_channel))
    app.add_handler(CommandHandler("stats", stats_cmd))

    # editing commands
    app.add_handler(CommandHandler("speed", speed_cmd))
    app.add_handler(CommandHandler("pitch", pitch_cmd))
    app.add_handler(CommandHandler("trim", trim_cmd))
    app.add_handler(CommandHandler("convert", convert_cmd))

    # preview/apply/post
    app.add_handler(CommandHandler("preview", preview_cmd))
    app.add_handler(CommandHandler("apply", apply_cmd))
    app.add_handler(CommandHandler("post_to_channel", post_to_channel_cmd))

    # photo handler (for album art + channel-forward trick)
    app.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, photo_handler))

    # audio upload handler (stores last_uploaded_audio)
    app.add_handler(MessageHandler((filters.VOICE | filters.AUDIO | (filters.Document & filters.Document.file_mime_type("audio/.*"))),
                                  handle_audio_upload))

    print("Starting Music Editor Full Bot...")
    app.run_polling()

if __name__ == "__main__":
    main()

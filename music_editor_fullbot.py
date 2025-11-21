# music_editor_fullbot.py
"""
Music Editor Full Bot
- Requires: python-telegram-bot >= 20.5 (tested with 22.5), aiofiles, ffmpeg on PATH
- Set BOT_TOKEN env var before running.
- Recommended to run as a Background Worker on Render (app.run_polling()).
"""

import os
import json
import asyncio
import subprocess
from pathlib import Path
from tempfile import mkstemp
from typing import Optional, Tuple
import time

from telegram import Update, InputFile
from telegram.error import TimedOut, TelegramError
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)

# ---------------- CONFIG ----------------
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
SETTINGS_FILE = "bot_settings.json"
STATS_FILE = "bot_stats.json"

MAX_FILE_SIZE_BYTES = 30 * 1024 * 1024
CONCURRENT_JOBS = 2
JOB_TIMEOUT_SECONDS = 90
job_semaphore = asyncio.Semaphore(CONCURRENT_JOBS)

# local test image path (optional)
TEST_LOCAL_IMAGE = "/mnt/data/e49ec989-1709-467d-992b-3944189f155c.png"

# ---------------- JSON HELPERS ----------------
def load_json(path, default):
    try:
        with open(path, "r", encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    try:
        with open(path, "w", encoding="utf8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("save_json error:", e, flush=True)


settings = load_json(SETTINGS_FILE, {"chats": {}, "channels": {}})
stats = load_json(STATS_FILE, {"total_processed": 0, "by_user": {}})


def ensure_chat_settings(chat_id):
    ks = str(chat_id)
    if ks not in settings["chats"]:
        settings["chats"][ks] = {
            "artist": None,
            "song_title": None,
            "picture_file_id": None,
            "last_edit_params": {}
        }
    return settings["chats"][ks]


# ---------------- UTILITIES ----------------
async def save_file_from_telegram(file_obj, dest_path: str):
    """Download Telegram File to dest_path, compatible across PTB versions."""
    f = await file_obj.get_file()
    if hasattr(f, "download_to_drive"):
        await f.download_to_drive(dest_path)
    else:
        await f.download(dest_path)
    return dest_path


def ffmpeg_run(cmd: list, timeout: int = JOB_TIMEOUT_SECONDS) -> Tuple[int, bytes, bytes]:
    """Synchronous ffmpeg run (safe to call in executor)."""
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except:
            pass
        return -1, b"", b"timeout"
    except Exception as e:
        return -2, b"", str(e).encode()


def build_atempo_filters(factor: float) -> str:
    if factor <= 0:
        factor = 1.0
    if 0.5 <= factor <= 2.0:
        return f"atempo={factor}"
    parts = []
    rem = factor
    while rem > 2.0:
        parts.append("atempo=2.0")
        rem /= 2.0
    while rem < 0.5:
        parts.append("atempo=0.5")
        rem /= 0.5
    parts.append(f"atempo={rem}")
    return ",".join(parts)


def build_pitch_filter(semitones: float) -> Optional[str]:
    if semitones == 0:
        return None
    factor = 2 ** (semitones / 12.0)
    return f"asetrate=44100*{factor},aresample=44100"


def format_time_arg(value: str) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    try:
        float(value)
        return value
    except:
        return value


# ---------------- TELEGRAM RETRY HELPER ----------------
async def _retry_telegram_call(coro_func, *args, retries: int = 2, delay: float = 1.0, **kwargs):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return await coro_func(*args, **kwargs)
        except (TimedOut, TelegramError, asyncio.TimeoutError) as e:
            last_exc = e
            if attempt < retries:
                await asyncio.sleep(delay * (attempt + 1))
                continue
            raise
    if last_exc:
        raise last_exc


# ---------------- COMMAND HANDLERS ----------------
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await _retry_telegram_call(update.message.reply_text, "Music Editor Bot ready. Use /help for commands.")


async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    txt = (
        "Commands:\n"
        "/setartist <name>\n"
        "/settitle <song name>\n"
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
        "Other:\n"
        "/stats\n"
    )
    await _retry_telegram_call(update.message.reply_text, txt)


async def set_artist(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await _retry_telegram_call(update.message.reply_text, "Usage: /setartist <artist name>")
        return
    cs = ensure_chat_settings(update.effective_chat.id)
    cs["artist"] = " ".join(ctx.args).strip()
    save_json(SETTINGS_FILE, settings)
    await _retry_telegram_call(update.message.reply_text, f"Artist set to: {cs['artist']}")


async def set_title(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await _retry_telegram_call(update.message.reply_text, "Usage: /settitle <song name>")
        return
    cs = ensure_chat_settings(update.effective_chat.id)
    cs["song_title"] = " ".join(ctx.args).strip()
    save_json(SETTINGS_FILE, settings)
    await _retry_telegram_call(update.message.reply_text, f"Song title set to: {cs['song_title']}")


async def set_pic_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data['awaiting_pic_for_chat'] = update.effective_chat.id
    await _retry_telegram_call(update.message.reply_text, "Send the photo you want as album art (photo or image file).")


async def connect_channel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if ctx.args:
        alias = ctx.args[0].strip()
        try:
            chat = await ctx.bot.get_chat(alias)
        except Exception as e:
            await _retry_telegram_call(update.message.reply_text, "Cannot find channel.")
            return
        settings["channels"][str(chat.id)] = {"title": chat.title, "username": chat.username}
        save_json(SETTINGS_FILE, settings)
        await _retry_telegram_call(update.message.reply_text, f"Channel connected: {chat.title}")
        return
    ctx.user_data['awaiting_channel_connect'] = True
    await _retry_telegram_call(update.message.reply_text, "Forward any channel post here to connect it.")


async def stats_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    t = stats.get("total_processed", 0)
    by_user = stats.get("by_user", {})
    top = sorted(by_user.items(), key=lambda x: -x[1])[:10]
    lines = [f"Total processed audios: {t}", "Top chats:"]
    for uid, cnt in top:
        lines.append(f"- {uid}: {cnt}")
    await _retry_telegram_call(update.message.reply_text, "\n".join(lines) if lines else "No stats yet.")


# ---------------- EDIT PARAMS ----------------
def set_last_params_for_chat(chat_id, params: dict):
    cs = ensure_chat_settings(chat_id)
    cs["last_edit_params"] = params
    save_json(SETTINGS_FILE, settings)


def get_last_params_for_chat(chat_id) -> dict:
    cs = ensure_chat_settings(chat_id)
    return cs.get("last_edit_params", {})


async def speed_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await _retry_telegram_call(update.message.reply_text, "Usage: /speed <factor>")
        return
    try:
        f = float(ctx.args[0])
    except:
        await _retry_telegram_call(update.message.reply_text, "Invalid number.")
        return
    params = get_last_params_for_chat(update.effective_chat.id)
    params["speed"] = f
    set_last_params_for_chat(update.effective_chat.id, params)
    await _retry_telegram_call(update.message.reply_text, f"Speed set: {f}")


async def pitch_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await _retry_telegram_call(update.message.reply_text, "Usage: /pitch <semitones>")
        return
    try:
        s = float(ctx.args[0])
    except:
        await _retry_telegram_call(update.message.reply_text, "Invalid number.")
        return
    params = get_last_params_for_chat(update.effective_chat.id)
    params["pitch"] = s
    set_last_params_for_chat(update.effective_chat.id, params)
    await _retry_telegram_call(update.message.reply_text, f"Pitch set: {s}")


async def trim_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) < 2:
        await _retry_telegram_call(update.message.reply_text, "Usage: /trim <start> <end>")
        return
    start = format_time_arg(ctx.args[0])
    end = format_time_arg(ctx.args[1])
    params = get_last_params_for_chat(update.effective_chat.id)
    params["trim_start"] = start
    params["trim_end"] = end
    set_last_params_for_chat(update.effective_chat.id, params)
    await _retry_telegram_call(update.message.reply_text, f"Trim set: {start} -> {end}")


async def convert_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await _retry_telegram_call(update.message.reply_text, "Usage: /convert <format>")
        return
    fmt = ctx.args[0].lower().strip()
    params = get_last_params_for_chat(update.effective_chat.id)
    params["convert_to"] = fmt
    set_last_params_for_chat(update.effective_chat.id, params)
    await _retry_telegram_call(update.message.reply_text, f"Format set: {fmt}")


# ---------------- PROCESSING ----------------
async def process_audio_file(in_path: str, out_path: str, params: dict) -> Tuple[bool, str]:
    filters_list = []
    if "speed" in params:
        try:
            filters_list.append(build_atempo_filters(float(params["speed"])))
        except:
            pass
    if "pitch" in params:
        try:
            pf = build_pitch_filter(float(params["pitch"]))
            if pf:
                filters_list.append(pf)
        except:
            pass
    filter_str = ",".join(filters_list) if filters_list else None

    cmd = ["ffmpeg", "-y"]
    if params.get("trim_start") is not None:
        cmd += ["-ss", str(params["trim_start"])]
    cmd += ["-i", in_path]
    if params.get("trim_end") is not None:
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


# ---------------- UPLOAD HANDLERS ----------------
async def handle_audio_upload(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        msg = update.message
        file_obj = None
        if msg.voice:
            file_obj = msg.voice
        elif msg.audio:
            file_obj = msg.audio
        elif msg.document and msg.document.mime_type and msg.document.mime_type.startswith("audio/"):
            file_obj = msg.document

        if not file_obj:
            await _retry_telegram_call(update.message.reply_text, "Send an audio file (voice, audio, or upload an audio file).")
            return

        fd, in_path = mkstemp()
        os.close(fd)
        await save_file_from_telegram(file_obj, in_path)

        if not file_size_ok(in_path):
            try:
                os.remove(in_path)
            except:
                pass
            await _retry_telegram_call(update.message.reply_text, "File too large. Limit ~30 MB.")
            return

        ctx.user_data["last_uploaded_audio"] = in_path
        await _retry_telegram_call(update.message.reply_text, "Audio received. Use /preview, /apply, or /post_to_channel.")
    except Exception as e:
        print("handle_audio_upload error:", e, flush=True)
        try:
            await _retry_telegram_call(update.message.reply_text, "Failed to process uploaded audio.")
        except:
            pass


async def preview_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        chat = update.effective_chat.id
        in_path = ctx.user_data.get("last_uploaded_audio")
        if not in_path or not Path(in_path).exists():
            await _retry_telegram_call(update.message.reply_text, "Send audio first.")
            return

        params = get_last_params_for_chat(chat).copy()
        if params.get("trim_start") is None:
            params["trim_start"] = 0
        params["trim_end"] = 10

        fd, out_path = mkstemp(suffix=".mp3")
        os.close(fd)
        await _retry_telegram_call(update.message.reply_text, "Generating preview...")

        async with job_semaphore:
            ok, err = await process_audio_file(in_path, out_path, params)

        if not ok:
            await _retry_telegram_call(update.message.reply_text, f"Preview failed: {err}")
            try: os.remove(out_path)
            except: pass
            return

        cs = ensure_chat_settings(chat)
        performer = cs.get("artist")
        title = cs.get("song_title")
        thumb = cs.get("picture_file_id")

        with open(out_path, "rb") as f:
            if thumb:
                await _retry_telegram_call(ctx.bot.send_audio, chat_id=chat, audio=InputFile(f, filename="preview.mp3"), thumb=thumb, performer=performer, title=title)
            else:
                await _retry_telegram_call(ctx.bot.send_audio, chat_id=chat, audio=InputFile(f, filename="preview.mp3"), performer=performer, title=title)
        try: os.remove(out_path)
        except: pass
    except Exception as e:
        print("preview_cmd error:", e, flush=True)
        try:
            await _retry_telegram_call(update.message.reply_text, "Preview failed due to internal error.")
        except:
            pass


async def apply_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        chat = update.effective_chat.id
        in_path = ctx.user_data.get("last_uploaded_audio")
        if not in_path or not Path(in_path).exists():
            await _retry_telegram_call(update.message.reply_text, "Send audio first.")
            return

        params = get_last_params_for_chat(chat)
        fmt = params.get("convert_to", "mp3")
        fd, out_path = mkstemp(suffix=f".{fmt}")
        os.close(fd)
        await _retry_telegram_call(update.message.reply_text, "Processing full file...")

        async with job_semaphore:
            ok, err = await process_audio_file(in_path, out_path, params)

        if not ok:
            await _retry_telegram_call(update.message.reply_text, f"Processing failed: {err}")
            try: os.remove(out_path)
            except: pass
            return

        # update stats
        stats["total_processed"] = stats.get("total_processed", 0) + 1
        uid = str(chat)
        stats.setdefault("by_user", {})
        stats["by_user"][uid] = stats["by_user"].get(uid, 0) + 1
        save_json(STATS_FILE, stats)

        cs = ensure_chat_settings(chat)
        performer = cs.get("artist")
        title = cs.get("song_title")
        thumb = cs.get("picture_file_id")

        with open(out_path, "rb") as f:
            if thumb:
                await _retry_telegram_call(ctx.bot.send_audio, chat_id=chat, audio=InputFile(f, filename=f"edited.{fmt}"), thumb=thumb, performer=performer, title=title)
            else:
                await _retry_telegram_call(ctx.bot.send_audio, chat_id=chat, audio=InputFile(f, filename=f"edited.{fmt}"), performer=performer, title=title)
        try: os.remove(out_path)
        except: pass
    except Exception as e:
        print("apply_cmd error:", e, flush=True)
        try:
            await _retry_telegram_call(update.message.reply_text, "Processing failed due to internal error.")
        except:
            pass


async def post_to_channel_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        chat = update.effective_chat.id
        in_path = ctx.user_data.get("last_uploaded_audio")
        if not in_path or not Path(in_path).exists():
            await _retry_telegram_call(update.message.reply_text, "Send audio first.")
            return
        if not settings.get("channels"):
            await _retry_telegram_call(update.message.reply_text, "No channels connected.")
            return

        channel_id = list(settings["channels"].keys())[0]
        params = get_last_params_for_chat(chat)
        fmt = params.get("convert_to", "mp3")
        fd, out_path = mkstemp(suffix=f".{fmt}")
        os.close(fd)
        await _retry_telegram_call(update.message.reply_text, "Posting to channel...")
        async with job_semaphore:
            ok, err = await process_audio_file(in_path, out_path, params)
        if not ok:
            await _retry_telegram_call(update.message.reply_text, f"Failed: {err}")
            try: os.remove(out_path)
            except: pass
            return

        cs = ensure_chat_settings(chat)
        performer = cs.get("artist")
        title = cs.get("song_title")
        thumb = cs.get("picture_file_id")

        with open(out_path, "rb") as f:
            await _retry_telegram_call(ctx.bot.send_audio, chat_id=int(channel_id), audio=InputFile(f, filename=f"channel_post.{fmt}"), thumb=thumb, performer=performer, title=title)
        await _retry_telegram_call(update.message.reply_text, f"Posted to channel (id: {channel_id}).")
        try: os.remove(out_path)
        except: pass
    except Exception as e:
        print("post_to_channel_cmd error:", e, flush=True)
        try:
            await _retry_telegram_call(update.message.reply_text, "Posting failed due to internal error.")
        except:
            pass


# ---------------- DOCUMENT ROUTER (safe) ----------------
async def document_router(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    Generic router for messages that may contain document attachments.
    We only handle:
    - image/* documents (save as album art if awaiting)
    - audio/* documents (pass to audio handler)
    Other documents are ignored.
    This avoids relying on filters.Document (incompatible across PTB versions).
    """
    try:
        msg = update.message
        if not msg:
            return

        # if it's a document...
        doc = msg.document
        if not doc:
            return

        mim = doc.mime_type or ""
        if mim.startswith("image/"):
            # route to photo_handler path that expects update with document
            # reuse photo_handler logic by calling it
            await photo_handler(update, ctx)
            return
        if mim.startswith("audio/"):
            await handle_audio_upload(update, ctx)
            return
    except Exception as e:
        print("document_router error:", e, flush=True)


# ---------------- UPLOAD LOCAL THUMB (testing) ----------------
async def upload_local_thumb_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(TEST_LOCAL_IMAGE):
        await _retry_telegram_call(update.message.reply_text, f"Local test image not found at {TEST_LOCAL_IMAGE}")
        return
    try:
        with open(TEST_LOCAL_IMAGE, "rb") as f:
            sent = await _retry_telegram_call(ctx.bot.send_photo, chat_id=update.effective_chat.id, photo=f)
        fid = sent.photo[-1].file_id
        cs = ensure_chat_settings(update.effective_chat.id)
        cs["picture_file_id"] = fid
        save_json(SETTINGS_FILE, settings)
        try:
            await ctx.bot.delete_message(chat_id=update.effective_chat.id, message_id=sent.message_id)
        except:
            pass
        await _retry_telegram_call(update.message.reply_text, "Local test image uploaded and saved as album art.")
    except Exception as e:
        print("upload_local_thumb_cmd error:", e, flush=True)
        await _retry_telegram_call(update.message.reply_text, f"Failed to upload local image: {e}")


# ---------------- WIRING & MAIN ----------------
def main():
    if not BOT_TOKEN:
        print("ERROR: BOT_TOKEN not set. Exiting.", flush=True)
        return

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("setartist", set_artist))
    app.add_handler(CommandHandler("settitle", set_title))
    app.add_handler(CommandHandler("setpic", set_pic_command))
    app.add_handler(CommandHandler("connect_channel", connect_channel))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("upload_local_thumb", upload_local_thumb_cmd))

    # editing
    app.add_handler(CommandHandler("speed", speed_cmd))
    app.add_handler(CommandHandler("pitch", pitch_cmd))
    app.add_handler(CommandHandler("trim", trim_cmd))
    app.add_handler(CommandHandler("convert", convert_cmd))

    # processing
    app.add_handler(CommandHandler("preview", preview_cmd))
    app.add_handler(CommandHandler("apply", apply_cmd))
    app.add_handler(CommandHandler("post_to_channel", post_to_channel_cmd))

    # media handlers: register specific ones first
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio_upload))
    # generic document router (will inspect mime and call appropriate routines)
    app.add_handler(MessageHandler(filters.ALL, document_router))

    print("Starting Music Editor Full Bot...", flush=True)
    try:
        app.run_polling()
    except Exception as e:
        print("Fatal error in run_polling:", e, flush=True)


if __name__ == "__main__":
    main()

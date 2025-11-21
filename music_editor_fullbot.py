# music_editor_fullbot_final.py
# Final cleaned-up Music Editor Bot
# Requirements (in requirements.txt):
# python-telegram-bot==22.5
# aiofiles
# Pillow
# ffmpeg available on PATH in the container

import os
import json
import asyncio
import subprocess
from pathlib import Path
from tempfile import mkstemp
from typing import Optional, Tuple
import io
import traceback

import aiofiles
from PIL import Image

from telegram import Update, InputFile
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)

# --- Config ---
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
SETTINGS_FILE = "bot_settings.json"
STATS_FILE = "bot_stats.json"

MAX_FILE_SIZE_BYTES = 30 * 1024 * 1024  # 30 MB
CONCURRENT_JOBS = 2
JOB_TIMEOUT_SECONDS = 120
job_semaphore = asyncio.Semaphore(CONCURRENT_JOBS)

# --- Simple JSON helpers ---

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
    except Exception:
        pass


settings = load_json(SETTINGS_FILE, {"chats": {}, "channels": {}})
stats = load_json(STATS_FILE, {"total_processed": 0, "by_user": {}})


def ensure_chat_settings(chat_id):
    ks = str(chat_id)
    if ks not in settings["chats"]:
        settings["chats"][ks] = {
            "artist": None,
            "title": None,
            "picture_file_id": None,
            "last_edit_params": {}
        }
    return settings["chats"][ks]


# --- Utilities ---
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
    except Exception:
        return value


# --- Image helper: produce JPEG bytes suitable for Telegram audio thumb ---
def prepare_image_for_thumb_bytes(local_path: str, max_size_px: int = 320, max_bytes: int = 200_000) -> bytes:
    try:
        img = Image.open(local_path).convert("RGB")
    except Exception:
        raise
    img.thumbnail((max_size_px, max_size_px), Image.LANCZOS)

    quality = 95
    last_data = b""
    for _ in range(8):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        last_data = data
        if len(data) <= max_bytes or quality <= 30:
            return data
        quality = max(30, quality - 10)
    return last_data


# --- Basic bot commands ---
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Music Editor Bot ready. Use /help for commands.")


async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    txt = (
        "Commands:\n"
        "/setartist <name> — set artist metadata\n"
        "/settitle <name> — set title metadata\n"
        "/setpic — send next image to register album art\n"
        "/connect_channel — forward any message from your channel after running this\n\n"
        "Edit params:\n"
        "/speed <factor> — e.g. /speed 1.25\n"
        "/pitch <semitones> — e.g. /pitch -2\n"
        "/trim <start> <end> — seconds or HH:MM:SS\n"
        "/convert <format> — mp3/wav/ogg/m4a\n\n"
        "Then send audio and use /preview or /apply.\n"
        "/post_to_channel — process & post to connected channel\n"
        "/stats — show processing stats\n"
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


async def set_title(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Usage: /settitle <song title>")
        return
    cs = ensure_chat_settings(update.effective_chat.id)
    cs["title"] = " ".join(ctx.args).strip()
    save_json(SETTINGS_FILE, settings)
    await update.message.reply_text(f"Title set to: {cs['title']}")


async def set_pic_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data['awaiting_pic_for_chat'] = update.effective_chat.id
    await update.message.reply_text("Send the photo you want to use as album art (any size/type). I'll convert & store it.")


async def stats_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    total = stats.get("total_processed", 0)
    await update.message.reply_text(f"Total processed audios: {total}")


# --- Editing parameter commands ---
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
    except Exception:
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
    except Exception:
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
    await update.message.reply_text(f"Trim set: {start} -> {end}")


async def convert_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Usage: /convert <format>")
        return
    fmt = ctx.args[0].lower().strip()
    params = get_last_params_for_chat(update.effective_chat.id)
    params["convert_to"] = fmt
    set_last_params_for_chat(update.effective_chat.id, params)
    await update.message.reply_text(f"Format set: {fmt}")


# --- Core process function ---
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


# --- File size check ---
def file_size_ok(path: str) -> bool:
    try:
        return Path(path).stat().st_size <= MAX_FILE_SIZE_BYTES
    except Exception:
        return False


# --- Audio upload handlers ---
async def handle_audio_upload(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    file_obj = msg.voice or msg.audio or (msg.document if msg.document and msg.document.mime_type and msg.document.mime_type.startswith("audio/") else None)
    if not file_obj:
        await msg.reply_text("Send an audio file or voice note.")
        return

    fd_in, in_path = mkstemp(suffix=".ogg")
    os.close(fd_in)
    try:
        await save_file_from_telegram(file_obj, in_path)
    except Exception:
        await msg.reply_text("Failed to download file.")
        try:
            os.remove(in_path)
        except:
            pass
        return

    if not file_size_ok(in_path):
        await msg.reply_text("File too large. Limit is ~30 MB.")
        try:
            os.remove(in_path)
        except:
            pass
        return

    # store path for this user/chat
    ctx.user_data["last_uploaded_audio"] = in_path
    await msg.reply_text("Audio received. Use /preview to get a short preview, or /apply to process the full file.")


async def preview_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    in_path = ctx.user_data.get("last_uploaded_audio")
    if not in_path or not Path(in_path).exists():
        await update.message.reply_text("No uploaded audio found. Send audio first.")
        return
    params = get_last_params_for_chat(chat_id).copy()
    params_preview = params.copy()
    if not params_preview.get("trim_start"):
        params_preview["trim_start"] = 0
    params_preview["trim_end"] = 10

    out_fd, out_path = mkstemp(suffix=".mp3")
    os.close(out_fd)

    await update.message.reply_text("Processing preview...")

    async with job_semaphore:
        ok, err = await process_audio_file(in_path, out_path, params_preview)
    if not ok:
        await update.message.reply_text(f"Preview failed: {err}")
        try:
            os.remove(out_path)
        except:
            pass
        return

    cs = ensure_chat_settings(chat_id)
    performer = cs.get("artist")
    title = cs.get("title")
    thumb = cs.get("picture_file_id")

    try:
        with open(out_path, "rb") as f:
            # Try to send with thumb/file_id — if the library wrapper doesn't accept thumb, fallback
            try:
                await ctx.bot.send_audio(chat_id=chat_id, audio=InputFile(f, filename="preview.mp3"), performer=performer or None, title=title or None, thumb=thumb if thumb else None)
            except TypeError:
                # some wrappers raise unexpected kwarg 'thumb' — try without thumb
                f.seek(0)
                await ctx.bot.send_audio(chat_id=chat_id, audio=InputFile(f, filename="preview.mp3"), performer=performer or None, title=title or None)
    except Exception as e:
        await update.message.reply_text("Failed to send preview; sending as document.")
        try:
            await ctx.bot.send_document(chat_id=chat_id, document=InputFile(out_path, filename="preview.mp3"))
        except Exception:
            pass
    finally:
        try:
            os.remove(out_path)
        except:
            pass


async def apply_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    in_path = ctx.user_data.get("last_uploaded_audio")
    if not in_path or not Path(in_path).exists():
        await update.message.reply_text("No uploaded audio found. Send audio first.")
        return
    params = get_last_params_for_chat(chat_id) or {}
    fmt = params.get("convert_to", "mp3")
    suffix = f".{fmt if fmt else 'mp3'}"
    out_fd, out_path = mkstemp(suffix=suffix)
    os.close(out_fd)

    await update.message.reply_text("Processing full file... This may take a moment.")

    async with job_semaphore:
        ok, err = await process_audio_file(in_path, out_path, params)
    if not ok:
        await update.message.reply_text(f"Processing failed: {err}")
        try:
            os.remove(out_path)
        except:
            pass
        return

    # update stats
    stats["total_processed"] = stats.get("total_processed", 0) + 1
    uid = str(chat_id)
    stats.setdefault("by_user", {})
    stats["by_user"][uid] = stats["by_user"].get(uid, 0) + 1
    save_json(STATS_FILE, stats)

    cs = ensure_chat_settings(chat_id)
    performer = cs.get("artist")
    title = cs.get("title")
    thumb = cs.get("picture_file_id")

    try:
        with open(out_path, "rb") as f:
            try:
                await ctx.bot.send_audio(chat_id=chat_id, audio=InputFile(f, filename=f"edited{suffix}"), performer=performer or None, title=title or None, thumb=thumb if thumb else None)
            except TypeError:
                f.seek(0)
                await ctx.bot.send_audio(chat_id=chat_id, audio=InputFile(f, filename=f"edited{suffix}"), performer=performer or None, title=title or None)
    except Exception:
        try:
            await ctx.bot.send_document(chat_id=chat_id, document=InputFile(out_path, filename=f"edited{suffix}"))
        except Exception:
            pass
    finally:
        try:
            os.remove(out_path)
        except:
            pass


async def post_to_channel_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    in_path = ctx.user_data.get("last_uploaded_audio")
    if not in_path or not Path(in_path).exists():
        await update.message.reply_text("No uploaded audio found. Send audio first.")
        return
    if not settings.get("channels"):
        await update.message.reply_text("No channels registered. Use /connect_channel first.")
        return
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
        try:
            os.remove(out_path)
        except:
            pass
        return

    cs = ensure_chat_settings(chat_id)
    performer = cs.get("artist")
    title = cs.get("title")
    thumb = cs.get("picture_file_id")

    try:
        with open(out_path, "rb") as f:
            try:
                await ctx.bot.send_audio(chat_id=int(channel_id), audio=InputFile(f, filename=f"channel_post{suffix}"), performer=performer or None, title=title or None, thumb=thumb if thumb else None)
            except TypeError:
                f.seek(0)
                await ctx.bot.send_audio(chat_id=int(channel_id), audio=InputFile(f, filename=f"channel_post{suffix}"), performer=performer or None, title=title or None)
        await update.message.reply_text(f"Posted to channel (id: {channel_id}).")
    except Exception as e:
        await update.message.reply_text(f"Failed to post to channel: {e}")
    finally:
        try:
            os.remove(out_path)
        except:
            pass


# --- Photo / Document handlers ---
async def photo_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    # Handles photo messages (telegram photo sizes) and image documents
    try:
        awaiting_chat = ctx.user_data.get('awaiting_pic_for_chat')
        # Photo message
        if update.message.photo:
            fd, tmp_path = mkstemp(suffix=".jpg")
            os.close(fd)
            try:
                file = await update.message.photo[-1].get_file()
                await file.download_to_drive(tmp_path)
                jpeg_bytes = prepare_image_for_thumb_bytes(tmp_path)
            finally:
                try: os.remove(tmp_path)
                except: pass

            sent = await ctx.bot.send_photo(chat_id=update.effective_chat.id, photo=io.BytesIO(jpeg_bytes))
            file_id = sent.photo[-1].file_id

            target_chat_id = awaiting_chat if (awaiting_chat and awaiting_chat == update.effective_chat.id) else update.effective_chat.id
            cs = ensure_chat_settings(target_chat_id)
            cs["picture_file_id"] = file_id
            save_json(SETTINGS_FILE, settings)
            ctx.user_data.pop('awaiting_pic_for_chat', None)
            await update.message.reply_text("Album art saved (converted to JPEG thumbnail).")
            return

        # Document that's an image
        if update.message.document and update.message.document.mime_type and update.message.document.mime_type.startswith("image/"):
            fd, tmp_path = mkstemp()
            os.close(fd)
            try:
                await save_file_from_telegram(update.message.document, tmp_path)
                jpeg_bytes = prepare_image_for_thumb_bytes(tmp_path)
            finally:
                try: os.remove(tmp_path)
                except: pass

            sent = await ctx.bot.send_photo(chat_id=update.effective_chat.id, photo=io.BytesIO(jpeg_bytes))
            file_id = sent.photo[-1].file_id

            target_chat_id = awaiting_chat if (awaiting_chat and awaiting_chat == update.effective_chat.id) else update.effective_chat.id
            cs = ensure_chat_settings(target_chat_id)
            cs["picture_file_id"] = file_id
            save_json(SETTINGS_FILE, settings)
            ctx.user_data.pop('awaiting_pic_for_chat', None)
            await update.message.reply_text("Album art saved (converted to JPEG thumbnail).")
            return

        # Channel-forward registration
        awaiting_channel = ctx.user_data.get('awaiting_channel_connect')
        if awaiting_channel and update.message.forward_from_chat and update.message.forward_from_chat.type == "channel":
            channel = update.message.forward_from_chat
            settings["channels"][str(channel.id)] = {"title": channel.title, "username": channel.username}
            save_json(SETTINGS_FILE, settings)
            ctx.user_data.pop('awaiting_channel_connect', None)
            await update.message.reply_text(f"Channel registered: {channel.title} (id: {channel.id})")
            return

    except Exception:
        traceback.print_exc()
        try:
            await update.message.reply_text("Failed to save album art (internal error).")
        except:
            pass


async def handle_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    # Documents may be images or audio files (or videos). Route appropriately.
    doc = update.message.document
    if not doc:
        return
    mt = doc.mime_type or ""
    if mt.startswith("image/"):
        await photo_handler(update, ctx)
        return
    if mt.startswith("audio/"):
        # treat as audio upload
        # turn document into msg.document and call audio handler
        await handle_audio_upload(update, ctx)
        return
    if mt.startswith("video/"):
        # try to accept video by extracting audio — simple: download and let ffmpeg pick audio
        fd, tmp = mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            await save_file_from_telegram(doc, tmp)
            # store as last_uploaded_audio (ffmpeg will extract audio during processing)
            ctx.user_data["last_uploaded_audio"] = tmp
            await update.message.reply_text("Video received; audio will be extracted when processing. Use /preview or /apply.")
        except Exception:
            try: os.remove(tmp)
            except: pass
            await update.message.reply_text("Failed to download video.")
        return


# --- Local test helper to register a pre-uploaded image in container ---
async def upload_local_thumb_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    default_path = "/mnt/data/09261ec4-eb57-4ed9-8d5a-ee01d752c0ba.png"
    path = ctx.args[0] if ctx.args else default_path
    if not os.path.exists(path):
        await update.message.reply_text(f"Local file not found: {path}")
        return
    try:
        jpeg_bytes = prepare_image_for_thumb_bytes(path)
        sent = await ctx.bot.send_photo(chat_id=update.effective_chat.id, photo=io.BytesIO(jpeg_bytes))
        file_id = sent.photo[-1].file_id
        cs = ensure_chat_settings(update.effective_chat.id)
        cs["picture_file_id"] = file_id
        save_json(SETTINGS_FILE, settings)
        await update.message.reply_text("Local file registered as album art (id saved).")
    except Exception:
        traceback.print_exc()
        await update.message.reply_text("Failed to register local file.")


# --- Wiring & main ---

def main():
    if not BOT_TOKEN:
        print("ERROR: BOT_TOKEN not set. Set BOT_TOKEN environment variable and restart.")
        return

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # core commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("setartist", set_artist))
    app.add_handler(CommandHandler("settitle", set_title))
    app.add_handler(CommandHandler("setpic", set_pic_command))
    app.add_handler(CommandHandler("connect_channel", connect_channel) if 'connect_channel' in globals() else CommandHandler("connect_channel", lambda u, c: c.bot.send_message(chat_id=u.effective_chat.id, text='connect_channel missing')))
    app.add_handler(CommandHandler("stats", stats_cmd))

    # params
    app.add_handler(CommandHandler("speed", speed_cmd))
    app.add_handler(CommandHandler("pitch", pitch_cmd))
    app.add_handler(CommandHandler("trim", trim_cmd))
    app.add_handler(CommandHandler("convert", convert_cmd))

    # processing commands
    app.add_handler(CommandHandler("preview", preview_cmd))
    app.add_handler(CommandHandler("apply", apply_cmd))
    app.add_handler(CommandHandler("post_to_channel", post_to_channel_cmd))
    app.add_handler(CommandHandler("upload_local_thumb", upload_local_thumb_cmd))

    # media handlers: separate handlers to avoid filter merging quirks
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(filters.Document, handle_document))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio_upload))

    print("Starting Music Editor Full Bot...")
    app.run_polling()


if __name__ == "__main__":
    main()

# music_editor_fullbot.py
# Requirements (example):
#   pip install python-telegram-bot==22.5 Pillow mutagen aiofiles python-dotenv
# Ensure ffmpeg is installed and on PATH.
#
# Usage:
#   set BOT_TOKEN env var, then: python music_editor_fullbot.py

import os
import json
import asyncio
import subprocess
from pathlib import Path
from tempfile import mkstemp, NamedTemporaryFile
from typing import Optional, Tuple
import io

import aiofiles
from PIL import Image
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC, TIT2, TPE1

from telegram import Update, InputFile
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
)

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
SETTINGS_FILE = "bot_settings.json"
STATS_FILE = "bot_stats.json"

MAX_FILE_SIZE_BYTES = 30 * 1024 * 1024  # 30 MB limit for processing environment
CONCURRENT_JOBS = 2
JOB_TIMEOUT_SECONDS = 120

job_semaphore = asyncio.Semaphore(CONCURRENT_JOBS)


# ---------------- JSON helpers ----------------
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


# ---------------- utilities ----------------
async def download_telegram_file(bot, file_id: str, dst_path: str) -> str:
    """Download a Telegram file by file_id to dst_path (uses telegram get_file)."""
    tf = await bot.get_file(file_id)
    await tf.download_to_drive(dst_path)
    return dst_path


def ffmpeg_run(cmd: list, timeout: int = JOB_TIMEOUT_SECONDS) -> Tuple[int, bytes, bytes]:
    """Synchronous ffmpeg call used inside executor."""
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
    # asetrate trick (will change duration; combine with atempo if desired)
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


# ---------------- thumbnail helpers (convert & embed) ----------------
def resize_and_convert_to_jpeg_bytes(in_path: str, max_size=(320, 320), target_kb=180) -> bytes:
    """
    Resize image to max_size, convert to JPEG and try to keep under target_kb.
    Returns JPEG bytes.
    """
    img = Image.open(in_path).convert("RGB")
    img.thumbnail(max_size, Image.LANCZOS)

    # binary search JPEG quality
    q_low, q_high = 20, 95
    best = None
    while q_low <= q_high:
        q_mid = (q_low + q_high) // 2
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q_mid, optimize=True)
        size_kb = buf.tell() / 1024
        if size_kb > target_kb:
            q_high = q_mid - 1
        else:
            best = buf.getvalue()
            q_low = q_mid + 1
    if best is None:
        # fallback: save at lower quality
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=40, optimize=True)
        return buf.getvalue()
    return best


def embed_cover_in_mp3_bytes(mp3_path: str, jpeg_bytes: bytes, title: Optional[str] = None, artist: Optional[str] = None):
    """
    Embed JPEG bytes into MP3 file at mp3_path using ID3 APIC frame. Overwrites tags if necessary.
    """
    try:
        audio = MP3(mp3_path, ID3=ID3)
    except Exception:
        audio = MP3(mp3_path)
    try:
        if audio.tags is None:
            audio.add_tags()
    except Exception:
        pass

    # remove existing APIC frames
    try:
        audio.tags.delall("APIC")
    except Exception:
        pass

    apic = APIC(
        encoding=3,  # 3 is UTF-8
        mime="image/jpeg",
        type=3,  # front cover
        desc="cover",
        data=jpeg_bytes,
    )
    audio.tags.add(apic)

    if title:
        try:
            audio.tags.delall("TIT2")
        except Exception:
            pass
        audio.tags.add(TIT2(encoding=3, text=str(title)))
    if artist:
        try:
            audio.tags.delall("TPE1")
        except Exception:
            pass
        audio.tags.add(TPE1(encoding=3, text=str(artist)))

    audio.save(v2_version=3)


# ---------------- ffmpeg processing ----------------
async def process_audio_file(in_path: str, out_path: str, params: dict) -> Tuple[bool, str]:
    """
    Build ffmpeg command and run in executor. Returns (ok, err_text)
    params: speed, pitch, trim_start, trim_end, convert_to
    """
    filters = []
    if "speed" in params:
        try:
            filters.append(build_atempo_filters(float(params["speed"])))
        except:
            pass
    if "pitch" in params:
        try:
            pf = build_pitch_filter(float(params["pitch"]))
            if pf:
                filters.append(pf)
        except:
            pass
    filter_str = ",".join(filters) if filters else None

    cmd = ["ffmpeg", "-y"]
    if params.get("trim_start"):
        cmd += ["-ss", str(params["trim_start"])]
    cmd += ["-i", in_path]
    if params.get("trim_end"):
        cmd += ["-to", str(params["trim_end"])]
    if filter_str:
        cmd += ["-filter:a", filter_str]

    outfmt = params.get("convert_to", "mp3").lower()

    # choose options by format
    if outfmt == "wav":
        cmd += ["-vn", "-ac", "2", "-ar", "44100", out_path]
    elif outfmt == "ogg":
        cmd += ["-vn", "-acodec", "libvorbis", "-ar", "44100", out_path]
    elif outfmt in ("m4a", "aac"):
        cmd += ["-vn", "-c:a", "aac", "-b:a", "192k", out_path]
    else:
        # default mp3
        cmd += ["-vn", "-ac", "2", "-ar", "44100", "-b:a", "192k", out_path]

    loop = asyncio.get_event_loop()
    rc, out, err = await loop.run_in_executor(None, ffmpeg_run, cmd)
    if rc == 0:
        return True, ""
    else:
        return False, err.decode(errors="ignore")


def file_size_ok(path: str) -> bool:
    try:
        return Path(path).stat().st_size <= MAX_FILE_SIZE_BYTES
    except:
        return False


# ---------------- command handlers ----------------
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Music Editor Bot ready. Use /help for commands.")


async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    txt = (
        "Commands:\n"
        "/setartist <name>\n"
        "/settitle <song title>\n"
        "/setpic  — then send a photo to save as cover\n"
        "/connect_channel — then forward any message from channel or /connect_channel @username\n\n"
        "Editing commands (set before sending audio):\n"
        "/speed <factor>\n"
        "/pitch <semitones>\n"
        "/trim <start_seconds> <end_seconds>\n"
        "/convert <format>\n\n"
        "Then upload audio and use:\n"
        "/preview — short 10s preview\n"
        "/apply — process full file and return music (cover embedded)\n"
        "/post_to_channel — post to connected channel (bot must be admin)\n\n"
        "/stats\n"
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
    await update.message.reply_text("Send the photo you want as album art (any size). I'll resize & embed it.")


async def upload_local_thumb_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    # convenience: allow uploading an image file as document to set as thumb
    ctx.user_data['awaiting_pic_for_chat'] = update.effective_chat.id
    await update.message.reply_text("Send the image file (as document) to use as local thumbnail.")


async def connect_channel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if ctx.args:
        alias = ctx.args[0].strip()
        try:
            chat = await ctx.bot.get_chat(alias)
        except Exception:
            await update.message.reply_text("Cannot find channel by that username.")
            return
        settings["channels"][str(chat.id)] = {"title": chat.title, "username": chat.username}
        save_json(SETTINGS_FILE, settings)
        await update.message.reply_text(f"Channel connected: {chat.title} (id: {chat.id})")
        return

    ctx.user_data['awaiting_channel_connect'] = True
    await update.message.reply_text("Forward a message from the channel here to connect it (bot must be admin in channel).")


async def stats_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    total = stats.get("total_processed", 0)
    by_user = stats.get("by_user", {})
    lines = [f"Total processed audios: {total}", "Top chats:"]
    top = sorted(by_user.items(), key=lambda x: -x[1])[:10]
    for uid, cnt in top:
        lines.append(f"- {uid}: {cnt}")
    await update.message.reply_text("\n".join(lines))


# ---------------- editing param commands ----------------
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
        await update.message.reply_text("Usage: /trim <start_seconds> <end_seconds>")
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
    await update.message.reply_text(f"Target format set to: {fmt}")


# ---------------- audio handlers ----------------
async def handle_audio_upload_file(bot, file_obj, dest_path):
    """Helper: download telegram file object to dest_path asynchronously if possible."""
    # file_obj may be Message.entity (Voice/Audio/Document) which has get_file()
    tf = await file_obj.get_file()
    await tf.download_to_drive(dest_path)
    return dest_path


async def handle_audio_upload(update: Update, ctx: ContextTypes.DEFAULT_TYPE, file_obj):
    msg = update.message
    fd, in_path = mkstemp()
    os.close(fd)
    try:
        await handle_audio_upload_file(ctx.bot, file_obj, in_path)
    except Exception as e:
        try:
            os.remove(in_path)
        except: pass
        await msg.reply_text("Download failed.")
        return

    if not file_size_ok(in_path):
        try:
            os.remove(in_path)
        except: pass
        await msg.reply_text("File too large.")
        return

    ctx.user_data["last_uploaded_audio"] = in_path
    await msg.reply_text("Audio received. Use /preview or /apply to process.")


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
        try: os.remove(out_path)
        except: pass
        await update.message.reply_text(f"Preview failed: {err}")
        return

    cs = ensure_chat_settings(chat_id)
    performer = cs.get("artist")
    title = cs.get("title")
    thumb_file_id = cs.get("picture_file_id")

    # If thumbnail exists, download -> resize -> embed into mp3 before sending
    try:
        if thumb_file_id:
            tmp_thumb = NamedTemporaryFile(delete=False, suffix=".img")
            tmp_thumb.close()
            await download_telegram_file(ctx.bot, thumb_file_id, tmp_thumb.name)
            jpeg_bytes = resize_and_convert_to_jpeg_bytes(tmp_thumb.name)
            embed_cover_in_mp3_bytes(out_path, jpeg_bytes, title=title, artist=performer)
            try: os.remove(tmp_thumb.name)
            except: pass
    except Exception:
        # continue even if embedding fails
        pass

    # send preview file
    try:
        with open(out_path, "rb") as f:
            await ctx.bot.send_audio(chat_id=chat_id, audio=InputFile(f, filename="preview.mp3"), performer=performer, title=title)
    except Exception:
        # fallback to document
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
    fmt = params.get("convert_to", "mp3").lower()
    fd, out_path = mkstemp(suffix=f".{fmt}")
    os.close(fd)

    await update.message.reply_text("Processing full file...")

    async with job_semaphore:
        ok, err = await process_audio_file(in_path, out_path, params)

    if not ok:
        try: os.remove(out_path)
        except: pass
        await update.message.reply_text(f"Processing failed: {err}")
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
    thumb_file_id = cs.get("picture_file_id")

    # embed cover if present and output is mp3 (best support)
    try:
        if thumb_file_id and fmt == "mp3":
            tmp_thumb = NamedTemporaryFile(delete=False, suffix=".img")
            tmp_thumb.close()
            await download_telegram_file(ctx.bot, thumb_file_id, tmp_thumb.name)
            jpeg_bytes = resize_and_convert_to_jpeg_bytes(tmp_thumb.name)
            embed_cover_in_mp3_bytes(out_path, jpeg_bytes, title=title, artist=performer)
            try: os.remove(tmp_thumb.name)
            except: pass
    except Exception:
        pass

    # send final file
    try:
        with open(out_path, "rb") as f:
            # prefer send_audio (so Telegram marks it as music)
            await ctx.bot.send_audio(chat_id=chat_id, audio=InputFile(f, filename=f"edited.{fmt}"), performer=performer, title=title)
    except Exception:
        await ctx.bot.send_document(chat_id=chat_id, document=InputFile(out_path, filename=f"edited.{fmt}"))
    finally:
        try: os.remove(out_path)
        except: pass


async def post_to_channel_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    in_path = ctx.user_data.get("last_uploaded_audio")
    if not in_path or not Path(in_path).exists():
        await update.message.reply_text("Send audio first.")
        return
    if not settings.get("channels"):
        await update.message.reply_text("No channels registered. Use /connect_channel first.")
        return

    channel_id = next(iter(settings["channels"].keys()))
    params = get_last_params_for_chat(chat_id) or {}
    fmt = params.get("convert_to", "mp3").lower()
    fd, out_path = mkstemp(suffix=f".{fmt}")
    os.close(fd)

    await update.message.reply_text("Processing and posting to channel...")

    async with job_semaphore:
        ok, err = await process_audio_file(in_path, out_path, params)

    if not ok:
        try: os.remove(out_path)
        except: pass
        await update.message.reply_text(f"Processing failed: {err}")
        return

    cs = ensure_chat_settings(chat_id)
    performer = cs.get("artist")
    title = cs.get("title")
    thumb_file_id = cs.get("picture_file_id")

    # embed cover if mp3
    try:
        if thumb_file_id and fmt == "mp3":
            tmp_thumb = NamedTemporaryFile(delete=False, suffix=".img")
            tmp_thumb.close()
            await download_telegram_file(ctx.bot, thumb_file_id, tmp_thumb.name)
            jpeg_bytes = resize_and_convert_to_jpeg_bytes(tmp_thumb.name)
            embed_cover_in_mp3_bytes(out_path, jpeg_bytes, title=title, artist=performer)
            try: os.remove(tmp_thumb.name)
            except: pass
    except Exception:
        pass

    try:
        with open(out_path, "rb") as f:
            await ctx.bot.send_audio(chat_id=int(channel_id), audio=InputFile(f, filename=f"channel_post.{fmt}"), performer=performer, title=title)
        await update.message.reply_text(f"Posted to channel (id: {channel_id}).")
    except Exception as e:
        await update.message.reply_text(f"Failed to post to channel: {e}")
    finally:
        try: os.remove(out_path)
        except: pass


# ---------------- generic message handler (photo, audio, voice, document, forwarded channel) ----------------
async def generic_message_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return

    # 1) Photo for /setpic or channel forward
    awaiting_pic = ctx.user_data.get('awaiting_pic_for_chat')
    if msg.photo and awaiting_pic and awaiting_pic == update.effective_chat.id:
        # take highest-res photo
        photo = msg.photo[-1]
        file_id = photo.file_id
        cs = ensure_chat_settings(awaiting_pic)
        cs["picture_file_id"] = file_id
        save_json(SETTINGS_FILE, settings)
        ctx.user_data.pop('awaiting_pic_for_chat', None)
        await msg.reply_text("Album art saved (will be embedded into outputs).")
        return

    awaiting_channel = ctx.user_data.get('awaiting_channel_connect')
    if awaiting_channel and msg.forward_from_chat and msg.forward_from_chat.type == "channel":
        channel = msg.forward_from_chat
        settings["channels"][str(channel.id)] = {"title": channel.title, "username": channel.username}
        save_json(SETTINGS_FILE, settings)
        ctx.user_data.pop('awaiting_channel_connect', None)
        await msg.reply_text(f"Channel registered: {channel.title} (id: {channel.id})")
        return

    # 2) Document as image (if awaiting pic)
    if msg.document and ctx.user_data.get('awaiting_pic_for_chat') == update.effective_chat.id:
        # accept image documents (jpg/png) as thumb
        mime = msg.document.mime_type or ""
        if mime.startswith("image/"):
            file_id = msg.document.file_id
            cs = ensure_chat_settings(update.effective_chat.id)
            cs["picture_file_id"] = file_id
            save_json(SETTINGS_FILE, settings)
            ctx.user_data.pop('awaiting_pic_for_chat', None)
            await msg.reply_text("Album art saved from document.")
            return

    # 3) Audio / Voice / Document (audio)
    # prefer voice, audio, then document with audio mime
    if msg.voice:
        await handle_audio_upload(update, ctx, msg.voice)
        return
    if msg.audio:
        await handle_audio_upload(update, ctx, msg.audio)
        return
    if msg.document:
        mime = (msg.document.mime_type or "").lower()
        if mime.startswith("audio/") or mime.startswith("video/"):
            # handle audio/video as file (ffmpeg will extract)
            await handle_audio_upload(update, ctx, msg.document)
            return

    # 4) If message contains video (voice?), accept by video attribute - extract audio
    if msg.video:
        await handle_audio_upload(update, ctx, msg.video)
        return

    # otherwise ignore or send simple help
    # we don't spam; do nothing
    return


# ---------------- wiring & main ----------------
def main():
    if not BOT_TOKEN:
        print("ERROR: BOT_TOKEN not set.")
        return

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # basic commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("setartist", set_artist))
    app.add_handler(CommandHandler("settitle", set_title))
    app.add_handler(CommandHandler("setpic", set_pic_command))
    app.add_handler(CommandHandler("upload_local_thumb", upload_local_thumb_cmd))
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

    # Generic message handler that inspects the message and dispatches
    app.add_handler(MessageHandler(filters.ALL, generic_message_handler))

    print("Starting Music Editor Full Bot...")
    app.run_polling()


if __name__ == "__main__":
    main()

# music_editor_fullbot.py
# Requires: python-telegram-bot==22.5, aiofiles, Pillow, mutagen, ffmpeg on PATH
# Set BOT_TOKEN env var and run: python music_editor_fullbot.py

import os
import json
import asyncio
import subprocess
from pathlib import Path
from tempfile import mkstemp
from typing import Optional, Tuple
import aiofiles
import io
import traceback

from PIL import Image
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC, TIT2, TPE1

from telegram import Update, InputFile
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
SETTINGS_FILE = "bot_settings.json"
STATS_FILE = "bot_stats.json"

MAX_FILE_SIZE_BYTES = 30 * 1024 * 1024  # 30 MB
CONCURRENT_JOBS = 2
JOB_TIMEOUT_SECONDS = 120

job_semaphore = asyncio.Semaphore(CONCURRENT_JOBS)

# ----------------- JSON helpers -----------------
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

# ----------------- utilities -----------------
async def save_file_from_telegram(file_obj, dest_path: str):
    """Download Telegram File to dest_path."""
    f = await file_obj.get_file()
    await f.download_to_drive(dest_path)
    return dest_path

def ffmpeg_run(cmd: list, timeout: int = JOB_TIMEOUT_SECONDS) -> Tuple[int, bytes, bytes]:
    """Run ffmpeg command synchronously."""
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

# ----------------- image helpers (convert & compress) -----------------
def convert_image_to_jpeg_bytes(in_path, max_size=(320,320), target_kb=160) -> bytes:
    """Resize/convert input image to JPEG bytes <= target_kb (approx)."""
    img = Image.open(in_path).convert("RGB")
    img.thumbnail(max_size, Image.LANCZOS)
    q_low, q_high = 20, 95
    best = None
    while q_low <= q_high:
        q_mid = (q_low + q_high)//2
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q_mid, optimize=True)
        size_kb = buf.tell() / 1024
        if size_kb > target_kb:
            q_high = q_mid - 1
        else:
            best = buf.getvalue()
            q_low = q_mid + 1
    if best is None:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=40, optimize=True)
        return buf.getvalue()
    return best

def embed_cover_in_mp3(mp3_path, jpeg_bytes, title=None, artist=None):
    """Embed JPEG bytes as APIC into MP3 (ID3v2.3)."""
    try:
        audio = MP3(mp3_path, ID3=ID3)
    except Exception:
        audio = MP3(mp3_path)
    if audio.tags is None:
        try:
            audio.add_tags()
        except:
            pass
    try:
        audio.tags.delall("APIC")
    except Exception:
        pass
    apic = APIC(encoding=3, mime='image/jpeg', type=3, desc='cover', data=jpeg_bytes)
    audio.tags.add(apic)
    if title:
        try: audio.tags.delall("TIT2")
        except: pass
        audio.tags.add(TIT2(encoding=3, text=str(title)))
    if artist:
        try: audio.tags.delall("TPE1")
        except: pass
        audio.tags.add(TPE1(encoding=3, text=str(artist)))
    audio.save(v2_version=3)

def ensure_mp3_with_cover(in_path, out_mp3_path, cover_image_path=None, title=None, artist=None):
    """Convert any input to MP3 (libmp3lame), embed cover if provided, return path."""
    cmd = ["ffmpeg", "-y", "-i", in_path, "-vn", "-c:a", "libmp3lame", "-b:a", "192k", out_mp3_path]
    rc, out, err = ffmpeg_run(cmd, timeout=JOB_TIMEOUT_SECONDS)
    if rc != 0:
        raise RuntimeError("ffmpeg conversion to mp3 failed: " + err.decode(errors="ignore"))
    if cover_image_path:
        jpeg_bytes = convert_image_to_jpeg_bytes(cover_image_path)
        embed_cover_in_mp3(out_mp3_path, jpeg_bytes, title=title, artist=artist)
    return out_mp3_path

# ----------------- processing core -----------------
async def process_audio_file(in_path: str, out_path: str, params: dict) -> Tuple[bool, str]:
    """
    Build and run ffmpeg command according to params.
    Returns (ok, error_message).
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
    if outfmt == "wav":
        cmd += ["-vn", "-ac", "2", "-ar", "44100", out_path]
    elif outfmt == "ogg":
        cmd += ["-vn", "-acodec", "libvorbis", "-ar", "44100", out_path]
    elif outfmt in ("m4a", "aac"):
        cmd += ["-vn", "-c:a", "aac", "-b:a", "192k", out_path]
    else:
        cmd += ["-vn", "-c:a", "libmp3lame", "-b:a", "192k", out_path]

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

# ----------------- command handlers -----------------
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Music Editor Bot ready. Use /help for commands.")

async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    txt = (
        "Commands:\n"
        "/setartist <name> — set artist metadata\n"
        "/setpic — run then send a photo or upload an image file to save as album art\n"
        "/connect_channel — run then forward a channel message OR /connect_channel @channel\n\n"
        "Editing commands (set before sending audio):\n"
        "/speed <factor>\n"
        "/pitch <semitones>\n"
        "/trim <start_seconds> <end_seconds>\n"
        "/convert <format> — mp3/wav/ogg/m4a\n\n"
        "Flow: set params → send audio/voice/video/document → /preview or /apply\n"
        "/preview — 10s mp3 preview (with cover if set)\n"
        "/apply — full processed mp3 (cover embedded)\n"
        "/post_to_channel — post processed audio to connected channel\n"
        "/stats — usage statistics"
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
    await update.message.reply_text("Now send the photo (or upload an image file) you want to use as album art.")

# photo handler: triggers for photos only
async def photo_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    awaiting_chat = ctx.user_data.get('awaiting_pic_for_chat')
    if awaiting_chat and awaiting_chat == update.effective_chat.id:
        photo = msg.photo[-1]  # highest resolution
        file_id = photo.file_id
        cs = ensure_chat_settings(awaiting_chat)
        cs["picture_file_id"] = file_id
        save_json(SETTINGS_FILE, settings)
        ctx.user_data.pop('awaiting_pic_for_chat', None)
        await update.message.reply_text("Album art saved (photo).")
        return

    awaiting_channel = ctx.user_data.get('awaiting_channel_connect')
    if awaiting_channel and msg.forward_from_chat and msg.forward_from_chat.type == "channel":
        channel = msg.forward_from_chat
        settings["channels"][str(channel.id)] = {"title": channel.title, "username": channel.username}
        save_json(SETTINGS_FILE, settings)
        ctx.user_data.pop('awaiting_channel_connect', None)
        await update.message.reply_text(f"Channel registered: {channel.title}")
        return

# document handler: handles uploaded documents (image/audio/video)
async def document_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    doc = msg.document
    if not doc:
        return
    mime = (doc.mime_type or "").lower()
    awaiting_chat = ctx.user_data.get('awaiting_pic_for_chat')
    if awaiting_chat and awaiting_chat == update.effective_chat.id and mime.startswith("image/"):
        cs = ensure_chat_settings(awaiting_chat)
        cs["picture_file_id"] = doc.file_id
        save_json(SETTINGS_FILE, settings)
        ctx.user_data.pop('awaiting_pic_for_chat', None)
        await update.message.reply_text("Album art saved (uploaded image).")
        return

    awaiting_channel = ctx.user_data.get('awaiting_channel_connect')
    if awaiting_channel and msg.forward_from_chat and msg.forward_from_chat.type == "channel":
        channel = msg.forward_from_chat
        settings["channels"][str(channel.id)] = {"title": channel.title, "username": channel.username}
        save_json(SETTINGS_FILE, settings)
        ctx.user_data.pop('awaiting_channel_connect', None)
        await update.message.reply_text(f"Channel registered: {channel.title}")
        return

    # if document is audio/video -> treat as upload
    if mime.startswith("audio/") or mime.startswith("video/"):
        ctx.user_data["last_uploaded_doc"] = doc
        await handle_audio_upload(update, ctx)
        return

    await msg.reply_text("Unsupported document type. For album art use an image (jpg/png). For audio send voice/audio/video.")

async def connect_channel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if ctx.args:
        alias = ctx.args[0].strip()
        try:
            chat = await ctx.bot.get_chat(alias)
        except Exception:
            await update.message.reply_text("Cannot find channel.")
            return
        settings["channels"][str(chat.id)] = {"title": chat.title, "username": chat.username}
        save_json(SETTINGS_FILE, settings)
        await update.message.reply_text(f"Channel connected: {chat.title}")
        return
    ctx.user_data['awaiting_channel_connect'] = True
    await update.message.reply_text("Forward any message from the channel here to connect it (bot must be admin).")

async def stats_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    t = stats.get("total_processed", 0)
    by_user = stats.get("by_user", {})
    top = sorted(by_user.items(), key=lambda x: -x[1])[:10]
    lines = [f"Total processed audios: {t}", "Top chats:"]
    for uid, cnt in top:
        lines.append(f"- {uid}: {cnt}")
    await update.message.reply_text("\n".join(lines))

# parameter commands
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
        await update.message.reply_text("Usage: /convert <format> (mp3/wav/ogg/m4a)")
        return
    fmt = ctx.args[0].lower().strip()
    params = get_last_params_for_chat(update.effective_chat.id)
    params["convert_to"] = fmt
    set_last_params_for_chat(update.effective_chat.id, params)
    await update.message.reply_text(f"Target format set to: {fmt}")

# ----------------- audio upload + preview/apply -----------------
async def handle_audio_upload(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Accept voice/audio/document (audio/video) and save to temp file path."""
    msg = update.message
    file_obj = None
    if msg.voice:
        file_obj = msg.voice
    elif msg.audio:
        file_obj = msg.audio
    elif msg.document:
        mime = (msg.document.mime_type or "").lower()
        if mime.startswith("audio/") or mime.startswith("video/"):
            file_obj = msg.document
    if not file_obj and ctx.user_data.get("last_uploaded_doc"):
        file_obj = ctx.user_data.pop("last_uploaded_doc")

    if not file_obj:
        await msg.reply_text("Send an audio file, voice note, or upload a video (audio extracted).")
        return

    fd, in_path = mkstemp()
    os.close(fd)

    try:
        await save_file_from_telegram(file_obj, in_path)
    except Exception:
        await msg.reply_text("Failed to download file.")
        try: os.remove(in_path)
        except: pass
        return

    if not file_size_ok(in_path):
        await msg.reply_text("File too large (limit ~30 MB).")
        try: os.remove(in_path)
        except: pass
        return

    ctx.user_data["last_uploaded_audio"] = in_path
    await msg.reply_text("Audio received. Now use /preview (short) or /apply (full).")

async def _generate_and_send_mp3_with_cover(update: Update, ctx, in_path: str, params: dict, preview=False):
    chat = update.effective_chat.id
    cs = ensure_chat_settings(chat)
    artist = cs.get("artist")
    picture_file_id = cs.get("picture_file_id")

    cover_path = None
    if picture_file_id:
        try:
            fd, cover_path = mkstemp(suffix=".img")
            os.close(fd)
            tgfile = await ctx.bot.get_file(picture_file_id)
            await tgfile.download_to_drive(cover_path)
        except Exception:
            try: os.remove(cover_path)
            except: pass
            cover_path = None

    params_preview = params.copy()
    if preview:
        params_preview["trim_start"] = params_preview.get("trim_start", 0)
        params_preview["trim_end"] = params_preview.get("trim_end", 10)

    fd_out, out_mp3 = mkstemp(suffix=".mp3")
    os.close(fd_out)

    try:
        loop = asyncio.get_event_loop()
        def convert_job():
            return ensure_mp3_with_cover(in_path, out_mp3, cover_image_path=cover_path, title=None, artist=artist)
        await loop.run_in_executor(None, convert_job)

        with open(out_mp3, "rb") as f:
            await ctx.bot.send_audio(chat_id=chat, audio=InputFile(f, filename="edited.mp3"), performer=artist)
    finally:
        try: os.remove(out_mp3)
        except: pass
        if cover_path:
            try: os.remove(cover_path)
            except: pass

async def preview_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat.id
    in_path = ctx.user_data.get("last_uploaded_audio")
    if not in_path or not Path(in_path).exists():
        await update.message.reply_text("No uploaded audio found. Send audio first.")
        return
    params = get_last_params_for_chat(chat).copy()
    await update.message.reply_text("Processing preview (short)...")
    async with job_semaphore:
        try:
            await _generate_and_send_mp3_with_cover(update, ctx, in_path, params, preview=True)
        except Exception as e:
            await update.message.reply_text(f"Preview failed: {e}")
            traceback.print_exc()

async def apply_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat.id
    in_path = ctx.user_data.get("last_uploaded_audio")
    if not in_path or not Path(in_path).exists():
        await update.message.reply_text("No uploaded audio found. Send audio first.")
        return
    params = get_last_params_for_chat(chat).copy()
    await update.message.reply_text("Processing full file... This may take a moment.")
    async with job_semaphore:
        fmt = params.get("convert_to", "mp3").lower()
        fd_out, out_path = mkstemp(suffix=f".{fmt}")
        os.close(fd_out)
        try:
            ok, err = await process_audio_file(in_path, out_path, params)
            if not ok:
                await update.message.reply_text(f"Processing failed: {err}")
                try: os.remove(out_path)
                except: pass
                return

            stats["total_processed"] = stats.get("total_processed", 0) + 1
            uid = str(chat)
            stats.setdefault("by_user", {})
            stats["by_user"][uid] = stats["by_user"].get(uid, 0) + 1
            save_json(STATS_FILE, stats)

            cs = ensure_chat_settings(chat)
            artist = cs.get("artist")
            picture_file_id = cs.get("picture_file_id")
            cover_path = None
            if picture_file_id:
                try:
                    fd_cp, cover_path = mkstemp(suffix=".img")
                    os.close(fd_cp)
                    tgfile = await ctx.bot.get_file(picture_file_id)
                    await tgfile.download_to_drive(cover_path)
                except:
                    cover_path = None

            if fmt == "mp3":
                if cover_path:
                    try:
                        jpeg_bytes = convert_image_to_jpeg_bytes(cover_path)
                        embed_cover_in_mp3(out_path, jpeg_bytes, title=None, artist=artist)
                    except:
                        pass
                    finally:
                        if cover_path:
                            try: os.remove(cover_path)
                            except: pass
                with open(out_path, "rb") as f:
                    await ctx.bot.send_audio(chat_id=chat, audio=InputFile(f, filename=f"edited.{fmt}"), performer=artist)
            else:
                fd_mp3, out_mp3 = mkstemp(suffix=".mp3")
                os.close(fd_mp3)
                try:
                    if picture_file_id:
                        cover_path2 = None
                        try:
                            fd_cp, cover_path2 = mkstemp(suffix=".img")
                            os.close(fd_cp)
                            tgfile = await ctx.bot.get_file(picture_file_id)
                            await tgfile.download_to_drive(cover_path2)
                        except:
                            cover_path2 = None
                    else:
                        cover_path2 = None
                    await asyncio.get_event_loop().run_in_executor(None, ensure_mp3_with_cover, out_path, out_mp3, cover_path2, None, artist)
                    with open(out_mp3, "rb") as f:
                        await ctx.bot.send_audio(chat_id=chat, audio=InputFile(f, filename="edited.mp3"), performer=artist)
                    with open(out_path, "rb") as fd:
                        await ctx.bot.send_document(chat_id=chat, document=InputFile(fd, filename=f"edited.{fmt}"))
                finally:
                    try: os.remove(out_mp3)
                    except: pass
                    if cover_path:
                        try: os.remove(cover_path)
                        except: pass

        finally:
            try: os.remove(out_path)
            except: pass

async def post_to_channel_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat.id
    in_path = ctx.user_data.get("last_uploaded_audio")
    if not in_path or not Path(in_path).exists():
        await update.message.reply_text("Send audio first.")
        return
    if not settings.get("channels"):
        await update.message.reply_text("No channels registered. Use /connect_channel.")
        return
    channel_id = next(iter(settings["channels"].keys()))
    params = get_last_params_for_chat(chat).copy()
    fmt = params.get("convert_to", "mp3").lower()
    fd_out, out_path = mkstemp(suffix=f".{fmt}")
    os.close(fd_out)
    await update.message.reply_text("Processing and posting to connected channel...")

    async with job_semaphore:
        ok, err = await process_audio_file(in_path, out_path, params)
    if not ok:
        await update.message.reply_text(f"Processing failed: {err}")
        try: os.remove(out_path)
        except: pass
        return

    cs = ensure_chat_settings(chat)
    artist = cs.get("artist")
    picture_file_id = cs.get("picture_file_id")
    cover_path = None
    if picture_file_id:
        try:
            fd_cp, cover_path = mkstemp(suffix=".img")
            os.close(fd_cp)
            tgfile = await ctx.bot.get_file(picture_file_id)
            await tgfile.download_to_drive(cover_path)
        except:
            cover_path = None

    try:
        if fmt == "mp3":
            if cover_path:
                jpeg_bytes = convert_image_to_jpeg_bytes(cover_path)
                embed_cover_in_mp3(out_path, jpeg_bytes, title=None, artist=artist)
            with open(out_path, "rb") as f:
                await ctx.bot.send_audio(chat_id=int(channel_id), audio=InputFile(f, filename=f"channel_post.{fmt}"), performer=artist)
        else:
            fd_mp3, out_mp3 = mkstemp(suffix=".mp3")
            os.close(fd_mp3)
            try:
                await asyncio.get_event_loop().run_in_executor(None, ensure_mp3_with_cover, out_path, out_mp3, cover_path, None, artist)
                with open(out_mp3, "rb") as f:
                    await ctx.bot.send_audio(chat_id=int(channel_id), audio=InputFile(f, filename="channel_post.mp3"), performer=artist)
                with open(out_path, "rb") as f:
                    await ctx.bot.send_document(chat_id=int(channel_id), document=InputFile(f, filename=f"channel_post.{fmt}"))
            finally:
                try: os.remove(out_mp3)
                except: pass
    except Exception as e:
        await update.message.reply_text(f"Failed to post to channel: {e}")
    finally:
        try: os.remove(out_path)
        except: pass
        if cover_path:
            try: os.remove(cover_path)
            except: pass
    await update.message.reply_text(f"Posted to channel id: {channel_id}")

# ----------------- wiring & main -----------------
def main():
    if not BOT_TOKEN:
        print("ERROR: BOT_TOKEN not set.")
        return

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # basic commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("setartist", set_artist))
    app.add_handler(CommandHandler("setpic", set_pic_command))
    app.add_handler(CommandHandler("connect_channel", connect_channel))
    app.add_handler(CommandHandler("stats", stats_cmd))

    # parameter/editing commands
    app.add_handler(CommandHandler("speed", speed_cmd))
    app.add_handler(CommandHandler("pitch", pitch_cmd))
    app.add_handler(CommandHandler("trim", trim_cmd))
    app.add_handler(CommandHandler("convert", convert_cmd))

    # preview/apply/post
    app.add_handler(CommandHandler("preview", preview_cmd))
    app.add_handler(CommandHandler("apply", apply_cmd))
    app.add_handler(CommandHandler("post_to_channel", post_to_channel_cmd))

    # handlers: separate for PHOTO and DOCUMENT (do NOT combine with inverted filters to avoid PTB filter type errors)
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(filters.Document, document_handler))

    # voice and audio handler
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio_upload))

    print("Starting Music Editor Full Bot...")
    app.run_polling()

if __name__ == "__main__":
    main()

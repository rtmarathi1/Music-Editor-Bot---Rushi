# music_editor_fullbot.py
# Replace your existing file with this.
# Requirements:
#   pip install python-telegram-bot==22.5 Pillow mutagen aiofiles python-dotenv
# Also ensure ffmpeg is installed in the container (apt-get install -y ffmpeg)

import os
import json
import asyncio
import traceback
import subprocess
from tempfile import mkstemp
from pathlib import Path
import io
from typing import Optional, Tuple

from PIL import Image
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC, TIT2, TPE1

from telegram import Update, InputFile
from telegram.error import NetworkError
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
SETTINGS_FILE = "bot_settings.json"
STATS_FILE = "bot_stats.json"
MAX_FILE_SIZE_BYTES = 40 * 1024 * 1024  # 40 MB safety
CONCURRENT_JOBS = 2
JOB_TIMEOUT_SECONDS = 180

# concurrency
job_semaphore = asyncio.Semaphore(CONCURRENT_JOBS)

# load/save JSON helpers
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
stats = load_json(STATS_FILE, {"total_processed": 0, "by_chat": {}})

def ensure_chat_settings(chat_id):
    k = str(chat_id)
    if k not in settings["chats"]:
        settings["chats"][k] = {"artist": None, "title": None, "picture_file_id": None, "last_params": {}}
    return settings["chats"][k]

# ---------- safe Telegram API wrapper with retries ----------
async def tg_call_with_retries(func, *args, retries=5, initial_delay=0.6, **kwargs):
    delay = initial_delay
    for attempt in range(1, retries + 1):
        try:
            return await func(*args, **kwargs)
        except (NetworkError, Exception) as e:
            # catch httpx.ReadError wrapped in NetworkError and other transient exceptions
            err_name = type(e).__name__
            if attempt == retries:
                raise
            await asyncio.sleep(delay)
            delay *= 2

# ---------- ffmpeg wrapper ----------
def ffmpeg_run(cmd: list, timeout: int = JOB_TIMEOUT_SECONDS) -> Tuple[int, bytes, bytes]:
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        proc.kill()
        return -1, b"", b"ffmpeg timeout"
    except Exception as e:
        return -2, b"", str(e).encode()

# audio processing helpers
def build_atempo_filters(factor: float) -> str:
    if factor <= 0:
        factor = 1.0
    if 0.5 <= factor <= 2.0:
        return f"atempo={factor}"
    parts = []
    remaining = factor
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining}")
    return ",".join(parts)

def build_pitch_filter(semitones: float) -> Optional[str]:
    if semitones == 0:
        return None
    factor = 2 ** (semitones / 12.0)
    return f"asetrate=44100*{factor},aresample=44100"

def process_audio_ffmpeg(in_path: str, out_path: str, params: dict) -> Tuple[bool, str]:
    filters = []
    if params.get("speed"):
        try:
            filters.append(build_atempo_filters(float(params["speed"])))
        except:
            pass
    if params.get("pitch"):
        try:
            pf = build_pitch_filter(float(params["pitch"]))
            if pf:
                filters.append(pf)
        except:
            pass
    filter_str = ",".join(filters) if filters else None

    cmd = ["ffmpeg", "-y"]
    # trimming
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
    elif outfmt in ("m4a", "aac"):
        cmd += ["-vn", "-c:a", "aac", "-b:a", "192k", out_path]
    elif outfmt == "ogg":
        cmd += ["-vn", "-c:a", "libvorbis", "-ar", "44100", out_path]
    else:  # default mp3
        cmd += ["-vn", "-c:a", "libmp3lame", "-b:a", "192k", out_path]

    rc, out, err = ffmpeg_run(cmd)
    if rc != 0:
        return False, err.decode(errors="ignore")
    return True, ""

# ---------- image helpers ----------
def convert_image_to_jpeg_bytes(in_path, max_size=(600,600), target_kb=180):
    img = Image.open(in_path).convert("RGB")
    img.thumbnail(max_size, Image.LANCZOS)
    # binary search quality to approximate target_kb
    low, high = 20, 95
    best = None
    while low <= high:
        mid = (low + high) // 2
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=mid, optimize=True)
        size_kb = buf.tell() / 1024
        if size_kb > target_kb:
            high = mid - 1
        else:
            best = buf.getvalue()
            low = mid + 1
    if best is None:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70, optimize=True)
        return buf.getvalue()
    return best

def embed_cover_in_mp3(mp3_path, jpeg_bytes, title=None, artist=None):
    try:
        audio = MP3(mp3_path, ID3=ID3)
    except Exception:
        audio = MP3(mp3_path)
    if audio.tags is None:
        try:
            audio.add_tags()
        except: pass
    try:
        audio.tags.delall("APIC")
    except: pass
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
    # first convert to mp3 (if not already), then embed cover
    tmp_fd, tmp_mp3 = mkstemp(suffix=".mp3")
    os.close(tmp_fd)
    try:
        ok, err = process_audio_ffmpeg(in_path, tmp_mp3, {"convert_to":"mp3"})
        if not ok:
            raise RuntimeError("ffmpeg -> mp3 failed: " + err)
        if cover_image_path:
            jpeg = convert_image_to_jpeg_bytes(cover_image_path)
            embed_cover_in_mp3(tmp_mp3, jpeg, title=title, artist=artist)
        else:
            # still add title/artist if provided
            if title or artist:
                try:
                    audio = MP3(tmp_mp3, ID3=ID3)
                except:
                    audio = MP3(tmp_mp3)
                if audio.tags is None:
                    try: audio.add_tags()
                    except: pass
                if title:
                    try: audio.tags.delall("TIT2")
                    except: pass
                    audio.tags.add(TIT2(encoding=3, text=str(title)))
                if artist:
                    try: audio.tags.delall("TPE1")
                    except: pass
                    audio.tags.add(TPE1(encoding=3, text=str(artist)))
                audio.save(v2_version=3)
        # move tmp_mp3 to out path
        os.replace(tmp_mp3, out_mp3_path)
    finally:
        try: os.remove(tmp_mp3)
        except: pass

# ---------- core handlers ----------
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text="Music Editor bot ready. /help")

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "Commands:\n"
        "/setartist <name>\n"
        "/settitle <title>\n"
        "/setpic — then send photo or upload image file\n"
        "/preview — 10s preview of last uploaded audio\n"
        "/apply — full processed audio\n"
        "/speed <factor>, /pitch <semitones>, /trim <start> <end>, /convert <fmt>\n"
    )
    await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text=help_text)

async def cmd_setartist(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text="Usage: /setartist <name>")
        return
    cs = ensure_chat_settings(update.effective_chat.id)
    cs["artist"] = " ".join(ctx.args).strip()
    save_json(SETTINGS_FILE, settings)
    await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text=f"Artist set to: {cs['artist']}")

async def cmd_settitle(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text="Usage: /settitle <song title>")
        return
    cs = ensure_chat_settings(update.effective_chat.id)
    cs["title"] = " ".join(ctx.args).strip()
    save_json(SETTINGS_FILE, settings)
    await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text=f"Title set to: {cs['title']}")

async def cmd_setpic(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    # mark that we are waiting for an image from this chat (photo or image upload)
    ctx.user_data['awaiting_pic_for_chat'] = update.effective_chat.id
    await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text="Send the photo/file you want to use as album art now.")

async def cmd_speed(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text="Usage: /speed <factor>")
        return
    try:
        f = float(ctx.args[0])
    except:
        await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text="Invalid number.")
        return
    cs = ensure_chat_settings(update.effective_chat.id)
    lp = cs.get("last_params", {})
    lp["speed"] = f
    cs["last_params"] = lp
    save_json(SETTINGS_FILE, settings)
    await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text=f"Speed set to {f}")

async def cmd_pitch(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text="Usage: /pitch <semitones>")
        return
    try:
        s = float(ctx.args[0])
    except:
        await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text="Invalid number.")
        return
    cs = ensure_chat_settings(update.effective_chat.id)
    lp = cs.get("last_params", {})
    lp["pitch"] = s
    cs["last_params"] = lp
    save_json(SETTINGS_FILE, settings)
    await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text=f"Pitch set to {s}")

async def cmd_trim(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) < 2:
        await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text="Usage: /trim <start> <end> (seconds or HH:MM:SS)")
        return
    start, end = ctx.args[0], ctx.args[1]
    cs = ensure_chat_settings(update.effective_chat.id)
    lp = cs.get("last_params", {})
    lp["trim_start"] = start
    lp["trim_end"] = end
    cs["last_params"] = lp
    save_json(SETTINGS_FILE, settings)
    await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text=f"Trim set: {start} -> {end}")

async def cmd_convert(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text="Usage: /convert <mp3|wav|m4a|ogg>")
        return
    fmt = ctx.args[0].lower()
    cs = ensure_chat_settings(update.effective_chat.id)
    lp = cs.get("last_params", {})
    lp["convert_to"] = fmt
    cs["last_params"] = lp
    save_json(SETTINGS_FILE, settings)
    await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text=f"Target format: {fmt}")

# ---------- upload/save handlers ----------
async def save_file_from_telegram(file_obj, dest_path: str, ctx: ContextTypes.DEFAULT_TYPE):
    tf = await tg_call_with_retries(file_obj.get_file)
    await tg_call_with_retries(tf.download_to_drive, dest_path)

async def handle_audio_file_upload(update: Update, ctx: ContextTypes.DEFAULT_TYPE, file_obj):
    fd, temp_path = mkstemp()
    os.close(fd)
    try:
        await save_file_from_telegram(file_obj, temp_path, ctx)
    except Exception as e:
        try: os.remove(temp_path)
        except: pass
        await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text=f"Failed to download: {e}")
        return
    if Path(temp_path).stat().st_size > MAX_FILE_SIZE_BYTES:
        try: os.remove(temp_path)
        except: pass
        await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text="File too large.")
        return
    ctx.user_data["last_uploaded_audio"] = temp_path
    await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text="Audio saved. Use /preview or /apply.")

# preview/apply utilities
async def generate_and_send_mp3_with_cover(update: Update, ctx: ContextTypes.DEFAULT_TYPE, preview=False):
    chat = update.effective_chat.id
    in_path = ctx.user_data.get("last_uploaded_audio")
    if not in_path or not Path(in_path).exists():
        await tg_call_with_retries(ctx.bot.send_message, chat_id=chat, text="No audio uploaded. Send audio first.")
        return
    cs = ensure_chat_settings(chat)
    params = cs.get("last_params", {}).copy()
    if preview:
        # guarantee preview length 10s if not trimmed
        params.setdefault("trim_start", 0)
        params.setdefault("trim_end", 10)

    # processing slot
    async with job_semaphore:
        fd_out, out_tmp = mkstemp(suffix="."+params.get("convert_to","mp3"))
        os.close(fd_out)
        cover_path = None
        try:
            ok, err = await asyncio.get_event_loop().run_in_executor(None, process_audio_ffmpeg, in_path, out_tmp, params)
            if not ok:
                await tg_call_with_retries(ctx.bot.send_message, chat_id=chat, text=f"Processing failed: {err}")
                return

            # create MP3 with embedded cover and ID3 tags even if format isn't mp3 (we'll send mp3 audio for player display)
            fd_mp3, out_mp3 = mkstemp(suffix=".mp3")
            os.close(fd_mp3)
            try:
                # download cover if available
                if cs.get("picture_file_id"):
                    fd_c, cover_path = mkstemp(suffix=".img")
                    os.close(fd_c)
                    tgfile = await tg_call_with_retries(ctx.bot.get_file, cs["picture_file_id"])
                    await tg_call_with_retries(tgfile.download_to_drive, cover_path)
                title = cs.get("title")
                artist = cs.get("artist")
                # ensure mp3 with cover and tags
                await asyncio.get_event_loop().run_in_executor(None, ensure_mp3_with_cover, out_tmp, out_mp3, cover_path, title, artist)
                # send audio using send_audio with title/performer params so clients display them.
                with open(out_mp3, "rb") as f:
                    await tg_call_with_retries(ctx.bot.send_audio, chat_id=chat, audio=InputFile(f, filename="music.mp3"),
                                               performer=artist, title=title)
            finally:
                try: os.remove(out_mp3)
                except: pass
        finally:
            try: os.remove(out_tmp)
            except: pass
            if cover_path:
                try: os.remove(cover_path)
                except: pass

async def cmd_preview(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text="Generating preview...")
    try:
        await generate_and_send_mp3_with_cover(update, ctx, preview=True)
    except Exception as e:
        await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text=f"Preview failed: {e}")
        traceback.print_exc()

async def cmd_apply(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text="Processing full file... (this may take a while)")
    try:
        await generate_and_send_mp3_with_cover(update, ctx, preview=False)
        # stats
        chat = str(update.effective_chat.id)
        stats["total_processed"] = stats.get("total_processed", 0) + 1
        stats.setdefault("by_chat", {})
        stats["by_chat"][chat] = stats["by_chat"].get(chat, 0) + 1
        save_json(STATS_FILE, stats)
    except Exception as e:
        await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text=f"Apply failed: {e}")
        traceback.print_exc()

# message handler for photos/documents/audio/voice/video
async def generic_message_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return

    # if awaiting channel connect logic could be added here

    # photo as album art
    if msg.photo:
        awaiting = ctx.user_data.get('awaiting_pic_for_chat')
        if awaiting and awaiting == update.effective_chat.id:
            # take highest resolution
            photo = msg.photo[-1]
            cs = ensure_chat_settings(update.effective_chat.id)
            cs["picture_file_id"] = photo.file_id
            save_json(SETTINGS_FILE, settings)
            ctx.user_data.pop('awaiting_pic_for_chat', None)
            await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text="Album art saved (photo).")
            return
        # else ignore
        return

    # document (image upload / audio)
    if msg.document:
        mime = (msg.document.mime_type or "").lower()
        awaiting = ctx.user_data.get('awaiting_pic_for_chat')
        if awaiting and awaiting == update.effective_chat.id and mime.startswith("image/"):
            cs = ensure_chat_settings(update.effective_chat.id)
            cs["picture_file_id"] = msg.document.file_id
            save_json(SETTINGS_FILE, settings)
            ctx.user_data.pop('awaiting_pic_for_chat', None)
            await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text="Album art saved (uploaded image).")
            return
        # audio/document/video => treat as upload
        if mime.startswith("audio/") or mime.startswith("video/"):
            await handle_audio_file_upload(update, ctx, msg.document)
            return
        await tg_call_with_retries(ctx.bot.send_message, chat_id=update.effective_chat.id, text="Unsupported document type. Send image for album art or audio/video for audio.")
        return

    # audio
    if msg.audio:
        await handle_audio_file_upload(update, ctx, msg.audio)
        return

    # voice
    if msg.voice:
        await handle_audio_file_upload(update, ctx, msg.voice)
        return

    # video
    if msg.video:
        await handle_audio_file_upload(update, ctx, msg.video)
        return

# ---------- startup ----------
def main():
    if not BOT_TOKEN:
        print("BOT_TOKEN not set. Exiting.")
        return

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # command handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("setartist", cmd_setartist))
    app.add_handler(CommandHandler("settitle", cmd_settitle))
    app.add_handler(CommandHandler("setpic", cmd_setpic))
    app.add_handler(CommandHandler("speed", cmd_speed))
    app.add_handler(CommandHandler("pitch", cmd_pitch))
    app.add_handler(CommandHandler("trim", cmd_trim))
    app.add_handler(CommandHandler("convert", cmd_convert))
    app.add_handler(CommandHandler("preview", cmd_preview))
    app.add_handler(CommandHandler("apply", cmd_apply))

    # generic message handler (single handler to avoid filter combination issues)
    app.add_handler(MessageHandler(filters.ALL, generic_message_handler))

    # remove webhook to avoid getUpdates conflicts (best effort)
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(app.bot.delete_webhook(drop_pending_updates=True))
        print("Webhook deleted at startup (if existed).")
    except Exception as e:
        print("Warning: could not delete webhook:", e)

    print("Starting bot (polling)...")
    try:
        app.run_polling(drop_pending_updates=True)
    except Exception as e:
        print("Bot failed:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()

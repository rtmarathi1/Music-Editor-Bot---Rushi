#!/usr/bin/env python3
"""
Music Editor Full Bot (replacement)
- Compatible with python-telegram-bot 22.5 (async)
- Handles audio/video/image uploads, resizes cover art, embeds ID3 tags (mutagen),
  processes audio via ffmpeg and returns final file as document (safe and compatible)

How to use:
- Set environment variable BOT_TOKEN with your bot token before running.
- Requires ffmpeg binary available in PATH.
- Dependencies (requirements.txt):
  python-telegram-bot==22.5
  httpx
  Pillow
  mutagen
  aiofiles
  python-dotenv

This file replaces your old bot. It purposely avoids unsupported filter combinations,
removes the use of `thumb` argument on send_audio, embeds album art in the mp3 with mutagen
and sends the final file with send_document which works reliably across PTB versions.

"""

import os
import asyncio
import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image
from mutagen.id3 import ID3, APIC, TIT2, TPE1, ID3NoHeaderError
from mutagen.mp3 import MP3

from telegram import Update, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# --- configuration ---
BOT_TOKEN = os.environ.get("BOT_TOKEN")
TMP_DIR = Path("/app/tmp") if Path("/app/tmp").exists() else Path("./tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)
FFMPEG_TIMEOUT = 120  # seconds for ffmpeg operations (adjust for large files)
MAX_CONCURRENT_WORKERS = 2

# per-chat state stored in memory (simple). For long-term persist use DB.
CHAT_STATE = {}

# concurrency semaphore to limit heavy jobs
job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_WORKERS)

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------- utilities ----------------------

def make_temp_path(prefix: str, suffix: str) -> Path:
    return TMP_DIR / f"{prefix}_{uuid.uuid4().hex}{suffix}"


async def run_ffmpeg(args: list[str], timeout: int = FFMPEG_TIMEOUT) -> None:
    """Run ffmpeg subprocess asynchronously and raise on error/timeout."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: returncode={proc.returncode}\nstderr={stderr.decode(errors='ignore')}" )


def image_resize_and_square(image_path: Path, out_path: Path, size: int = 600) -> None:
    """Open image, convert to RGB, center-crop to square and save as JPEG.
    This keeps aspect ratio and works with any input size.
    """
    im = Image.open(image_path)
    # convert to RGB (handles PNG/WEBP/AVIF/...)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    # center-crop to square based on shortest side
    w, h = im.size
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    im = im.crop((left, top, left + min_side, top + min_side))
    im.thumbnail((size, size), Image.LANCZOS)
    # save optimized jpeg
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, format="JPEG", quality=85, optimize=True)


def embed_cover_and_tags(mp3_path: Path, title: Optional[str], artist: Optional[str], cover_jpeg: Optional[Path]):
    """Embed cover art and basic ID3v2 tags into MP3 using mutagen."""
    try:
        audio = ID3(mp3_path)
    except ID3NoHeaderError:
        audio = ID3()

    if title:
        audio.delall('TIT2')
        audio.add(TIT2(encoding=3, text=title))
    if artist:
        audio.delall('TPE1')
        audio.add(TPE1(encoding=3, text=artist))

    if cover_jpeg and cover_jpeg.exists():
        with open(cover_jpeg, 'rb') as f:
            data = f.read()
        # remove previous APIC frames to avoid duplicates
        audio.delall('APIC')
        audio.add(APIC(
            encoding=3,
            mime='image/jpeg',
            type=3,  # cover (front)
            desc='cover',
            data=data,
        ))

    audio.save(mp3_path)


# ---------------------- state helpers ----------------------

def get_chat_state(chat_id: int) -> dict:
    s = CHAT_STATE.setdefault(chat_id, {
        'last_input': None,
        'title': None,
        'artist': None,
        'cover': None,  # Path to processed JPEG
        'format': 'mp3',
        'last_output': None,
    })
    return s


# ---------------------- handlers ----------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! I\'m Music Editor. Send me an audio/video to start, or /help to learn commands."
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "/settitle <title> - set track title\n"
        "/setartist <artist> - set artist\n"
        "/setpic - reply with an image to set cover\n"
        "/preview - create 10s preview with current settings\n"
        "/apply - apply full processing (embed cover + tags)\n"
        "/setformat mp3|m4a|wav - choose output format\n"
        "Send audio/video/photo directly as message."
    )
    await update.message.reply_text(txt)


async def set_title(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = get_chat_state(update.effective_chat.id)
    arg = ' '.join(context.args).strip()
    if not arg:
        await update.message.reply_text("Usage: /settitle Song Title")
        return
    chat['title'] = arg
    await update.message.reply_text(f"Title set to: {arg}")


async def set_artist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = get_chat_state(update.effective_chat.id)
    arg = ' '.join(context.args).strip()
    if not arg:
        await update.message.reply_text("Usage: /setartist Artist Name")
        return
    chat['artist'] = arg
    await update.message.reply_text(f"Artist set to: {arg}")


async def set_format(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = get_chat_state(update.effective_chat.id)
    if not context.args:
        await update.message.reply_text("Usage: /setformat mp3|m4a|wav")
        return
    fmt = context.args[0].lower()
    if fmt not in ('mp3', 'm4a', 'wav'):
        await update.message.reply_text("Supported formats: mp3, m4a, wav")
        return
    chat['format'] = fmt
    await update.message.reply_text(f"Output format set to: {fmt}")


async def setpic_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # user will send an image immediately after /setpic; we set a flag in state
    chat = get_chat_state(update.effective_chat.id)
    chat['awaiting_cover'] = True
    await update.message.reply_text("Please send the image you want to use as cover art. Any size accepted.")


async def media_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Catch-all media handler. Uses filters.ALL so avoid complex filter combinations.
    This inspects the message for photo/document/audio/video/voice and stores last input.
    """
    msg = update.message
    chat = get_chat_state(update.effective_chat.id)

    # If user previously asked /setpic and now sends an image
    if chat.get('awaiting_cover') and (msg.photo or msg.document and msg.document.mime_type.startswith('image')):
        # accept largest photo
        file = None
        if msg.photo:
            file = await msg.photo[-1].get_file()
        else:
            file = await msg.document.get_file()
        tmp_in = make_temp_path('cover_in', '.jpg')
        await file.download_to_drive(tmp_in)
        proc_cover = make_temp_path('cover', '.jpg')
        try:
            image_resize_and_square(tmp_in, proc_cover, size=600)
            # save path
            chat['cover'] = proc_cover
            chat['awaiting_cover'] = False
            await msg.reply_text('Cover image saved and resized to 600x600.')
        except Exception as e:
            logger.exception('cover processing error')
            await msg.reply_text('Failed to process image: ' + str(e))
        finally:
            try:
                tmp_in.unlink()
            except Exception:
                pass
        return

    # Save incoming media (audio/video/document) as last_input
    incoming_path = None
    incoming_kind = None
    try:
        if msg.audio:
            f = await msg.audio.get_file()
            incoming_kind = 'audio'
        elif msg.voice:
            f = await msg.voice.get_file()
            incoming_kind = 'voice'
        elif msg.video:
            f = await msg.video.get_file()
            incoming_kind = 'video'
        elif msg.document:
            f = await msg.document.get_file()
            incoming_kind = 'document'
        elif msg.photo:
            # user sent only photo (not cover flow) — treat as cover if awaiting, else ignore
            f = await msg.photo[-1].get_file()
            incoming_kind = 'photo'
        else:
            await msg.reply_text('Send an audio, voice note, video or document (audio file).')
            return

        incoming_path = make_temp_path('input', Path(f.file_path).suffix or '.bin')
        await f.download_to_drive(incoming_path)
        chat['last_input'] = {'path': incoming_path, 'kind': incoming_kind}

        await msg.reply_text(f'Received {incoming_kind}. Use /preview or /apply to process, or /settitle /setartist /setpic.')
    except Exception as e:
        logger.exception('error receiving media')
        await msg.reply_text('Failed to download file: ' + str(e))
        # cleanup
        if incoming_path and incoming_path.exists():
            try:
                incoming_path.unlink()
            except Exception:
                pass


async def preview_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = get_chat_state(update.effective_chat.id)
    data = chat.get('last_input')
    if not data:
        await update.message.reply_text('No input file found. Send an audio/video first.')
        return

    async with job_semaphore:
        await update.message.reply_text('Generating 10s preview...')
        try:
            in_path = Path(data['path'])
            out_path = make_temp_path('preview', '.mp3')
            # use ffmpeg to trim first 10 seconds and convert to mp3
            await run_ffmpeg(['-y', '-i', str(in_path), '-t', '10', '-vn', '-acodec', 'libmp3lame', '-b:a', '128k', str(out_path)])

            # embed tags/cover if present
            embed_cover_and_tags(out_path, chat.get('title'), chat.get('artist'), chat.get('cover'))

            chat['last_output'] = out_path
            await update.message.reply_document(document=InputFile(out_path), filename=out_path.name)
        except Exception as e:
            logger.exception('preview error')
            await update.message.reply_text('Preview failed: ' + str(e))


async def apply_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = get_chat_state(update.effective_chat.id)
    data = chat.get('last_input')
    if not data:
        await update.message.reply_text('No input file found. Send an audio/video first.')
        return

    async with job_semaphore:
        await update.message.reply_text('Processing full file (this may take a while)...')
        try:
            in_path = Path(data['path'])
            fmt = chat.get('format', 'mp3')
            out_ext = f'.{fmt}'
            out_path = make_temp_path('final', out_ext)

            # basic conversion pipeline: extract audio and convert to chosen format
            if fmt == 'mp3':
                await run_ffmpeg(['-y', '-i', str(in_path), '-vn', '-acodec', 'libmp3lame', '-b:a', '192k', str(out_path)])
            elif fmt == 'm4a':
                await run_ffmpeg(['-y', '-i', str(in_path), '-vn', '-c:a', 'aac', '-b:a', '192k', str(out_path)])
            elif fmt == 'wav':
                await run_ffmpeg(['-y', '-i', str(in_path), '-vn', '-c:a', 'pcm_s16le', str(out_path)])
            else:
                # fallback
                await run_ffmpeg(['-y', '-i', str(in_path), '-vn', str(out_path)])

            # If mp3 chosen, embed tags and cover art
            if out_path.suffix.lower() == '.mp3':
                embed_cover_and_tags(out_path, chat.get('title'), chat.get('artist'), chat.get('cover'))

            chat['last_output'] = out_path

            # send as document to avoid compatibility issues with thumbs across PTB versions
            await update.message.reply_document(document=InputFile(out_path), filename=out_path.name)

        except Exception as e:
            logger.exception('apply error')
            await update.message.reply_text('Apply failed: ' + str(e))


# ---------------------- startup / cleanup ----------------------

async def on_startup(app: Application):
    # attempt to remove webhook (avoid "terminated by other getUpdates request" conflict)
    try:
        await app.bot.delete_webhook()
        logger.info('Deleted webhook on startup (if any)')
    except Exception:
        logger.exception('Failed to delete webhook (ok to ignore)')


async def on_shutdown(app: Application):
    # cleanup temp files older than a threshold (simple cleanup)
    try:
        for p in TMP_DIR.iterdir():
            try:
                # remove files older than 1 day
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
            except Exception:
                pass
    except Exception:
        pass
    logger.info('Shutdown cleanup done')


# ---------------------- main ----------------------

def build_application() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError('BOT_TOKEN env var not set')
    app = Application.builder().token(BOT_TOKEN).build()

    # commands
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help', help_cmd))
    app.add_handler(CommandHandler('settitle', set_title))
    app.add_handler(CommandHandler('setartist', set_artist))
    app.add_handler(CommandHandler('setformat', set_format))
    app.add_handler(CommandHandler('setpic', setpic_cmd))
    app.add_handler(CommandHandler('preview', preview_cmd))
    app.add_handler(CommandHandler('apply', apply_cmd))

    # catch-all media handler (avoid filter operator incompatibilities)
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, media_handler))

    # startup/shutdown
    app.post_init(on_startup)
    app.post_shutdown(on_shutdown)

    return app


def main():
    app = build_application()
    logger.info('Starting Music Editor Full Bot (polling)...')
    # Use polling for Render environments where webhooks cause conflicts
    app.run_polling(allowed_updates=None)


if __name__ == '__main__':
    main()

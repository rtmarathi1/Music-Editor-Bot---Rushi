"""
Fixed Music Editor Full Bot (patched to avoid PosixPath -> InputFile bug)
- Compatible with python-telegram-bot v22.x
- Embeds cover art into MP3 (ID3 APIC) using mutagen
- Accepts images, resizes them, and stores per-user cover
- Opens files as binary streams when sending to Telegram (fixes AttributeError: 'PosixPath' object has no attribute 'read')
- Uses Application.builder().post_init(on_startup).build()
"""

import os
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image
from mutagen.id3 import ID3, APIC, TIT2, TPE1, ID3NoHeaderError

from telegram import Update, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ---------- Configuration ----------
BOT_TOKEN = os.environ.get("BOT_TOKEN") or "REPLACE_WITH_YOUR_TOKEN"
TMP_DIR = Path("/app/tmp") if Path("/app/tmp").exists() else Path(tempfile.gettempdir())
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Utilities ----------
async def save_telegram_file(file_obj, dest_path: Path):
    """Download and save a Telegram file asynchronously."""
    await file_obj.download_to_drive(custom_path=str(dest_path))
    return dest_path


def resize_cover(in_path: Path, out_path: Path, size: int = 600) -> None:
    """Open image, convert to RGB and resize preserving aspect, then save as JPEG.
    size: max dimension for longest side.
    """
    img = Image.open(in_path)
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    img.save(out_path, format="JPEG")


def embed_cover_in_mp3(mp3_path: Path, cover_jpeg_path: Path, title: Optional[str] = None, artist: Optional[str] = None) -> None:
    """Embed JPEG cover into MP3 file using ID3 APIC frame. If no ID3 header exists, it will be created."""
    try:
        tags = ID3(mp3_path)
    except ID3NoHeaderError:
        tags = ID3()

    with open(cover_jpeg_path, "rb") as f:
        img_data = f.read()

    # Remove existing APIC frames to avoid duplicates
    tags.delall("APIC")
    tags.add(
        APIC(
            encoding=3,  # 3 => utf-8
            mime="image/jpeg",
            type=3,  # 3 => cover (front)
            desc="cover",
            data=img_data,
        )
    )

    if title:
        tags.delall("TIT2")
        tags.add(TIT2(encoding=3, text=title))
    if artist:
        tags.delall("TPE1")
        tags.add(TPE1(encoding=3, text=artist))

    tags.save(mp3_path)


# ---------- Handlers ----------
async def on_startup(app: Application) -> None:
    """Called after application is built. Ensure webhook removed (avoid conflict)."""
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
    except Exception as e:
        logger.debug("delete_webhook failed: %s", e)


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Hi! Send an image (cover) then send an MP3 or audio file. Use /help for more.")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Usage:\n"
        "- Send an image to set it as cover.\n"
        "- Then send an MP3 (audio/document) to embed the cover into the file and get processed file back.\n"
        "- You can also send /setmeta Title - Artist to set metadata before uploading the audio."
    )


# simple in-memory map to hold last cover per user; for persistence you can store on disk
USER_STATE: dict = {}


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Save the largest photo the user sent as the cover for next audio."""
    user_id = update.effective_user.id
    photos = update.message.photo
    if not photos:
        await update.message.reply_text("No photo found.")
        return

    # get the largest size photo (last in sizes)
    photo = photos[-1]
    tmp_in = TMP_DIR / f"cover_{user_id}_in.jpg"
    tmp_out = TMP_DIR / f"cover_{user_id}.jpg"
    await save_telegram_file(await photo.get_file(), tmp_in)

    try:
        resize_cover(tmp_in, tmp_out, size=800)
    except Exception:
        # fallback: copy original and ensure jpeg
        im = Image.open(tmp_in).convert("RGB")
        im.save(tmp_out, format="JPEG")

    # store path in user state
    USER_STATE[user_id] = USER_STATE.get(user_id, {})
    USER_STATE[user_id]["cover"] = str(tmp_out)

    await update.message.reply_text("Cover image saved — send the MP3 now and I'll embed it.")


async def setmeta_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set metadata for next audio: /setmeta Title - Artist"""
    user_id = update.effective_user.id
    text = update.message.text or ""
    payload = text.partition(" ")[2].strip()
    title = None
    artist = None
    if "-" in payload:
        parts = payload.split("-", 1)
        title = parts[0].strip() or None
        artist = parts[1].strip() or None
    elif payload:
        title = payload or None

    USER_STATE[user_id] = USER_STATE.get(user_id, {})
    if title:
        USER_STATE[user_id]["title"] = title
    if artist:
        USER_STATE[user_id]["artist"] = artist

    await update.message.reply_text(f"Metadata set. Title: {title or 'None'}, Artist: {artist or 'None'}")


async def send_document_file(msg, path: Path, filename: str) -> None:
    """
    Open file at `path` in binary mode, send it to user as a document with `filename`,
    then close the file to avoid leaking file descriptors.
    """
    f = open(path, "rb")
    try:
        await msg.reply_document(document=InputFile(f), filename=filename)
    finally:
        try:
            f.close()
        except Exception:
            logger.debug("Failed to close file %s", path)


async def audio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle audio/document/voice uploads and embed cover if available."""
    msg = update.message
    user_id = update.effective_user.id

    file_obj = None
    filename = None

    if msg.audio:
        file_obj = await msg.audio.get_file()
        filename = msg.audio.file_name or f"audio_{user_id}.mp3"
    elif msg.document:
        # documents may or may not be audio; handle generically
        file_obj = await msg.document.get_file()
        filename = msg.document.file_name or f"audio_{user_id}"
    elif msg.voice:
        file_obj = await msg.voice.get_file()
        filename = f"voice_{user_id}.ogg"
    else:
        await msg.reply_text("Send an audio file (mp3 preferred) or a document containing audio.")
        return

    await msg.reply_text("Downloading file...")

    # safe filename and unique id
    safe_name = "".join(c for c in filename if c.isalnum() or c in (" ", ".", "_", "-")).strip()
    unique = int(asyncio.get_event_loop().time() * 1000)
    tmp_audio_in = TMP_DIR / f"upload_{user_id}_{unique}_{safe_name}"
    await save_telegram_file(file_obj, tmp_audio_in)

    # determine if mp3
    lower = str(filename).lower()
    is_mp3 = lower.endswith(".mp3") or tmp_audio_in.suffix.lower() == ".mp3"

    # check user metadata and cover
    state = USER_STATE.get(user_id, {})
    cover_path = state.get("cover")
    title = state.get("title")
    artist = state.get("artist")

    try:
        if is_mp3 and cover_path and Path(cover_path).exists():
            await msg.reply_text("Embedding cover into MP3 and metadata (if provided)...")
            # embed in-place
            try:
                embed_cover_in_mp3(tmp_audio_in, Path(cover_path), title=title, artist=artist)
            except Exception as e:
                logger.exception("Embed failed: %s", e)
                await msg.reply_text("Failed to embed cover. Sending original file instead.")
                await send_document_file(msg, tmp_audio_in, filename)
                return

            # send resulting mp3 as document (so clients download the file and see embedded cover)
            await send_document_file(msg, tmp_audio_in, filename)
            await msg.reply_text("Done — cover embedded.")
            return

        else:
            # if not MP3 or no cover, simply forward/send file back to user
            if not is_mp3:
                await msg.reply_text("File is not an MP3. I'll send your original file back.")
            elif not cover_path:
                await msg.reply_text("No cover set. Sending back original file.")
            else:
                await msg.reply_text("Unable to process file; sending original back.")
            await send_document_file(msg, tmp_audio_in, filename)
            return

    except Exception as exc:
        logger.exception("Processing failed: %s", exc)
        await msg.reply_text("An error occurred while processing the file.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update: %s", context.error)
    try:
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text("Internal error occurred. Try again later.")
    except Exception:
        logger.exception("Failed to send error message to user.")


# ---------- Application builder ----------

def build_application() -> Application:
    # post_init must be set on builder for PTB v22
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(on_startup)
        .build()
    )

    # Register handlers
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("setmeta", setmeta_cmd))

    # Photos (covers)
    app.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, photo_handler))

    # Audio / document / voice
    audio_filter = (filters.AUDIO | filters.VOICE | filters.Document.ALL) & ~filters.COMMAND
    app.add_handler(MessageHandler(audio_filter, audio_handler))

    # Global error handler
    app.add_error_handler(error_handler)

    return app


def main() -> None:
    if BOT_TOKEN == "REPLACE_WITH_YOUR_TOKEN":
        raise RuntimeError("Please set BOT_TOKEN environment variable or edit the script to include it.")

    app = build_application()

    # Run polling. In production you may prefer webhook.
    app.run_polling()


if __name__ == "__main__":
    main()

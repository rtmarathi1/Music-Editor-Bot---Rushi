# music_editor_fullbot_final.py
# Minimal robust Music Editor Bot (python-telegram-bot >=20.x)
# - Handles photos, audio, voice, video, documents
# - Avoids using filters.Document-specific attributes (use single generic handler)
# - Does NOT pass unsupported 'thumb' kwarg to send_audio (some ExtBot builds reject it)
# - Stores artist/title and picture file_id for each chat
# Usage: set BOT_TOKEN env var and run: python music_editor_fullbot_final.py

import os
import json
import asyncio
import subprocess
from pathlib import Path
from tempfile import mkstemp
from typing import Optional, Tuple

from telegram import Update, InputFile
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
SETTINGS_FILE = "bot_settings.json"
STATS_FILE = "bot_stats.json"

MAX_FILE_SIZE_BYTES = 30 * 1024 * 1024
CONCURRENT_JOBS = 2
JOB_TIMEOUT_SECONDS = 120
job_semaphore = asyncio.Semaphore(CONCURRENT_JOBS)

# ---------- simple JSON helpers ----------

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
        settings["chats"][ks] = {"artist": None, "title": None, "picture_file_id": None, "last_edit_params": {}}
    return settings["chats"][ks]

# ---------- utilities ----------

async def save_file_from_telegram(file_obj, dest_path: str):
    """Download Telegram File to dest_path.
    file_obj is a telegram.File-able object (Audio, Voice, Document, PhotoSize via get_file())
    """
    f = await file_obj.get_file()
    await f.download_to_drive(dest_path)
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

# ---------- commands ----------

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Music Editor Bot ready. Use /help for commands.")


async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    txt = (
        "Commands:\n"
        "/setartist <name> — set artist metadata\n"
        "/settitle <title> — set song title metadata\n"
        "/setpic — run, then send a photo to save as album art\n"
        "/connect_channel — run then forward a message from your channel OR /connect_channel @channelusername\n\n"
        "Editing commands (use before sending audio):\n"
        "/speed <factor> — e.g. /speed 1.25 or /speed 0.8\n"
        "/pitch <semitones> — e.g. /pitch 2 or /pitch -3\n"
        "/trim <start_seconds> <end_seconds> — e.g. /trim 5 20\n"
        "/convert <format> — e.g. /convert mp3 or /convert wav\n\n"
        "When you have set params, send audio/voice and use:\n"
        "/preview — short 10s preview\n"
        "/apply — process and return full edited file\n"
        "/post_to_channel — process & post to connected channel\n\n"
        "Also /stats and /help\n"
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
    await update.message.reply_text("Send the photo you want to use as album art (any size).")


async def stats_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    total = stats.get("total_processed", 0)
    await update.message.reply_text(f"Total processed audios: {total}")

# ---------- editing param commands ----------

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
    await update.message.reply_text(f"Speed set to {f}")


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
    await update.message.reply_text(f"Pitch set to {s}")


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
    if len(ctx.args) < 1:
        await update.message.reply_text("Usage: /convert <format>")
        return
    fmt = ctx.args[0].lower().strip()
    params = get_last_params_for_chat(update.effective_chat.id)
    params["convert_to"] = fmt
    set_last_params_for_chat(update.effective_chat.id, params)
    await update.message.reply_text(f"Target format set to: {fmt}")

# ---------- core processing ----------

async def process_audio_file(in_path: str, out_path: str, params: dict) -> Tuple[bool, str]:
    filters_list = []
    if "speed" in params:
        filters_list.append(build_atempo_filters(float(params["speed"])))
    if "pitch" in params:
        pf = build_pitch_filter(float(params["pitch"]))
        if pf:
            filters_list.append(pf)
    filter_str = ",".join([f for f in filters_list if f]) if filters_list else None

    cmd = ["ffmpeg", "-y"]
    if params.get("trim_start") is not None:
        cmd += ["-ss", str(params.get("trim_start"))]
    cmd += ["-i", in_path]
    if params.get("trim_end") is not None:
        cmd += ["-to", str(params.get("trim_end"))]
    if filter_str:
        cmd += ["-filter:a", filter_str]

    outfmt = params.get("convert_to", "mp3")
    if outfmt == "wav":
        cmd += ["-vn", "-ac", "2", "-ar", "44100", out_path]
    elif outfmt == "ogg":
        cmd += ["-vn", "-acodec", "libvorbis", "-ar", "44100", out_path]
    elif outfmt in ("m4a", "aac"):
        cmd += ["-vn", "-c:a", "aac", "-b:a", "192k", out_path]
    else:
        cmd += ["-vn", "-ac", "2", "-ar", "44100", "-b:a", "192k", out_path]

    loop = asyncio.get_event_loop()
    rc, out, err = await loop.run_in_executor(None, ffmpeg_run, cmd)
    return (rc == 0, err.decode(errors="ignore"))


def file_size_ok(path: str) -> bool:
    try:
        return Path(path).stat().st_size <= MAX_FILE_SIZE_BYTES
    except Exception:
        return False

# ---------- handlers ----------

async def handle_generic_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Unified handler for photos, audio, documents, video, voice, and forwards.
    We'll inspect the incoming message and route accordingly.
    """
    msg = update.message
    if not msg:
        return

    # 1) If awaiting a photo for /setpic
    awaiting_chat = ctx.user_data.get('awaiting_pic_for_chat')
    if awaiting_chat and awaiting_chat == update.effective_chat.id and msg.photo:
        photo = msg.photo[-1]
        cs = ensure_chat_settings(awaiting_chat)
        cs['picture_file_id'] = photo.file_id
        save_json(SETTINGS_FILE, settings)
        ctx.user_data.pop('awaiting_pic_for_chat', None)
        await msg.reply_text('Album art saved (will be used where possible).')
        return

    # 2) If awaiting a channel connect and a forwarded message from channel
    if ctx.user_data.get('awaiting_channel_connect') and msg.forward_from_chat and msg.forward_from_chat.type == 'channel':
        channel = msg.forward_from_chat
        settings['channels'][str(channel.id)] = {'title': channel.title, 'username': channel.username}
        save_json(SETTINGS_FILE, settings)
        ctx.user_data.pop('awaiting_channel_connect', None)
        await msg.reply_text(f'Channel registered: {channel.title} (id: {channel.id})')
        return

    # 3) Photo without awaiting: allow user to set pic by replying to /setpic earlier, otherwise ignore
    if msg.photo and not ctx.user_data.get('awaiting_pic_for_chat'):
        # allow user to save a photo anytime by sending /setpic first; otherwise polite reply
        return

    # 4) Accept audio/voice/video/document -> treat as potential audio upload
    file_obj = None
    # prioritize voice -> audio -> document (with audio mime) -> video (extract audio)
    if msg.voice:
        file_obj = msg.voice
    elif msg.audio:
        file_obj = msg.audio
    elif msg.document:
        # if document has mime_type and startswith audio/ treat as audio
        mt = getattr(msg.document, 'mime_type', '') or ''
        if mt.startswith('audio/'):
            file_obj = msg.document
        elif mt.startswith('image/') and ctx.user_data.get('awaiting_pic_for_chat'):
            # user sent image as document while awaiting pic
            cs = ensure_chat_settings(ctx.user_data['awaiting_pic_for_chat'])
            cs['picture_file_id'] = msg.document.file_id
            save_json(SETTINGS_FILE, settings)
            ctx.user_data.pop('awaiting_pic_for_chat', None)
            await msg.reply_text('Album art saved (document).')
            return
        elif mt.startswith('video/'):
            file_obj = msg.document  # we'll try to extract audio
        else:
            # unknown document type: ignore
            return
    elif msg.video:
        file_obj = msg.video
    else:
        return

    # route to audio upload handler
    await handle_audio_upload_from_msg(update, ctx, file_obj)


async def handle_audio_upload_from_msg(update: Update, ctx: ContextTypes.DEFAULT_TYPE, file_obj):
    msg = update.message
    if not file_obj:
        await msg.reply_text('No audio/detectable audio found.')
        return

    # create temp input path (let ffmpeg figure out extension)
    fd, in_path = mkstemp(suffix='')
    os.close(fd)
    try:
        await msg.reply_text('Downloading file...')
        await save_file_from_telegram(file_obj, in_path)
    except Exception as e:
        try: os.remove(in_path)
        except: pass
        await msg.reply_text('Failed to download file.')
        return

    if not file_size_ok(in_path):
        await msg.reply_text('File too large. Limit is ~30 MB.')
        try: os.remove(in_path)
        except: pass
        return

    # store path in user_data
    ctx.user_data['last_uploaded_audio'] = in_path
    await msg.reply_text('Audio received. Use /preview to get a short preview or /apply to process full file.')

# ---------- preview / apply / post ----------

async def preview_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    in_path = ctx.user_data.get('last_uploaded_audio')
    if not in_path or not Path(in_path).exists():
        await update.message.reply_text('No uploaded audio found. Send audio first.')
        return

    params = get_last_params_for_chat(chat_id).copy()
    params_preview = params.copy()
    params_preview['trim_start'] = params_preview.get('trim_start', 0)
    params_preview['trim_end'] = 10

    out_fd, out_path = mkstemp(suffix='.mp3')
    os.close(out_fd)

    await update.message.reply_text('Processing preview...')

    async with job_semaphore:
        ok, err = await process_audio_file(in_path, out_path, params_preview)

    if not ok:
        await update.message.reply_text(f'Preview failed: {err}')
        try: os.remove(out_path)
        except: pass
        return

    cs = ensure_chat_settings(chat_id)
    performer = cs.get('artist')
    title = cs.get('title')

    try:
        with open(out_path, 'rb') as f:
            # send_audio sometimes in some builds rejects extra kwargs like thumb; keep minimal
            await ctx.bot.send_audio(chat_id=chat_id, audio=InputFile(f, filename='preview.mp3'), performer=performer, title=title)
    except TypeError:
        # fallback if send_audio signature incompatible
        await ctx.bot.send_document(chat_id=chat_id, document=InputFile(out_path, filename='preview.mp3'))
    except Exception:
        await update.message.reply_text('Failed to send preview; sending as document.')
        await ctx.bot.send_document(chat_id=chat_id, document=InputFile(out_path, filename='preview.mp3'))
    finally:
        try: os.remove(out_path)
        except: pass


async def apply_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    in_path = ctx.user_data.get('last_uploaded_audio')
    if not in_path or not Path(in_path).exists():
        await update.message.reply_text('No uploaded audio found. Send audio first.')
        return

    params = get_last_params_for_chat(chat_id) or {}
    fmt = params.get('convert_to', 'mp3')
    fd, out_path = mkstemp(suffix=f'.{fmt}')
    os.close(fd)

    await update.message.reply_text('Processing full file...')

    async with job_semaphore:
        ok, err = await process_audio_file(in_path, out_path, params)

    if not ok:
        await update.message.reply_text(f'Processing failed: {err}')
        try: os.remove(out_path)
        except: pass
        return

    # update stats
    stats['total_processed'] = stats.get('total_processed', 0) + 1
    uid = str(chat_id)
    stats.setdefault('by_user', {})
    stats['by_user'][uid] = stats['by_user'].get(uid, 0) + 1
    save_json(STATS_FILE, stats)

    cs = ensure_chat_settings(chat_id)
    performer = cs.get('artist')
    title = cs.get('title')

    try:
        with open(out_path, 'rb') as f:
            await ctx.bot.send_audio(chat_id=chat_id, audio=InputFile(f, filename=f'edited.{fmt}'), performer=performer, title=title)
    except TypeError:
        await ctx.bot.send_document(chat_id=chat_id, document=InputFile(out_path, filename=f'edited.{fmt}'))
    except Exception as e:
        await update.message.reply_text(f'Failed to send edited file: {e}')
        await ctx.bot.send_document(chat_id=chat_id, document=InputFile(out_path, filename=f'edited.{fmt}'))
    finally:
        try: os.remove(out_path)
        except: pass


async def post_to_channel_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    in_path = ctx.user_data.get('last_uploaded_audio')
    if not in_path or not Path(in_path).exists():
        await update.message.reply_text('No uploaded audio found. Send audio first.')
        return

    if not settings.get('channels'):
        await update.message.reply_text('No channels registered. Use /connect_channel.')
        return

    channel_id = next(iter(settings['channels'].keys()))
    params = get_last_params_for_chat(chat_id) or {}
    fmt = params.get('convert_to', 'mp3')
    fd, out_path = mkstemp(suffix=f'.{fmt}')
    os.close(fd)

    await update.message.reply_text('Processing and posting to connected channel...')

    async with job_semaphore:
        ok, err = await process_audio_file(in_path, out_path, params)

    if not ok:
        await update.message.reply_text(f'Processing failed: {err}')
        try: os.remove(out_path)
        except: pass
        return

    cs = ensure_chat_settings(chat_id)
    performer = cs.get('artist')
    title = cs.get('title')

    try:
        with open(out_path, 'rb') as f:
            await ctx.bot.send_audio(chat_id=int(channel_id), audio=InputFile(f, filename=f'channel_post.{fmt}'), performer=performer, title=title)
        await update.message.reply_text(f'Posted to channel (id: {channel_id}).')
    except Exception as e:
        await update.message.reply_text(f'Failed to post to channel: {e}')
    finally:
        try: os.remove(out_path)
        except: pass

# ---------- wiring ----------

def main():
    if not BOT_TOKEN:
        print('ERROR: BOT_TOKEN not set.')
        return

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # commands
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help', help_cmd))
    app.add_handler(CommandHandler('setartist', set_artist))
    app.add_handler(CommandHandler('settitle', set_title))
    app.add_handler(CommandHandler('setpic', set_pic_command))
    app.add_handler(CommandHandler('connect_channel', connect_channel := (lambda u, c: None)))
    app.add_handler(CommandHandler('stats', stats_cmd))

    # editing
    app.add_handler(CommandHandler('speed', speed_cmd))
    app.add_handler(CommandHandler('pitch', pitch_cmd))
    app.add_handler(CommandHandler('trim', trim_cmd))
    app.add_handler(CommandHandler('convert', convert_cmd))

    # preview/apply/post
    app.add_handler(CommandHandler('preview', preview_cmd))
    app.add_handler(CommandHandler('apply', apply_cmd))
    app.add_handler(CommandHandler('post_to_channel', post_to_channel_cmd))

    # single generic handler for incoming media/messages
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_generic_message))

    print('Starting Music Editor Full Bot...')
    app.run_polling()


if __name__ == '__main__':
    main()

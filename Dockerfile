# Use an official slim Python image
FROM python:3.11-slim

# Install ffmpeg + system deps needed by pillow/ffmpeg
RUN apt-get update && apt-get install -y ffmpeg


RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ffmpeg \
      build-essential \
      libsndfile1 \
      libavcodec-extra \
      && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install python deps
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application
COPY . /app

# Ensure working dir has write access for temp files
RUN mkdir -p /app/tmp && chmod -R 777 /app/tmp

# ENV: optional runtime options
ENV PYTHONUNBUFFERED=1
# optional: set path to test image inside container (change if you add a different image)
ENV TEST_LOCAL_IMAGE="/mnt/data/e49ec989-1709-467d-992b-3944189f155c.png"

# Start the bot (works for Background Worker on Render)
CMD ["python", "music_editor_fullbot.py"]

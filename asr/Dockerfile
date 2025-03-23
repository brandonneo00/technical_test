# Use a lightweight Python base image
FROM python:3.13.2

# Install system dependencies needed by librosa (libsndfile for reading audio files, ffmpeg for mp3, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Copy your requirements.txt first so Docker can cache the pip install step
COPY requirements.txt /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your code (including asr_api.py) into /app
COPY . /app

# Expose port 8001 for the Flask service
EXPOSE 8001

# By default, run asr_api.py
CMD ["python", "asr_api.py"]

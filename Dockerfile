FROM python:3.11-slim

# Install system dependencies (ffmpeg and libsndfile1 for audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app/ ./app/

# Create a local cache directory for deepfilternet models (cached during runtime startup)
# DeepFilterNet downloads model weights (around 50MB) upon model init.
# We set the cache path environment variable so it persists or is contained.
ENV DEEPFILTER_CACHE_DIR=/root/.cache/deepfilter
RUN mkdir -p /root/.cache/deepfilter

# Expose port and run server
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

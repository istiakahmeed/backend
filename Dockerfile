FROM python:3.11-slim

# Install system dependencies (compiler, ffmpeg and libsndfile1 for audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set cache directory for DeepFilterNet (appdirs uses XDG_CACHE_HOME on Linux)
ENV XDG_CACHE_HOME=/app/.cache
RUN mkdir -p /app/.cache

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the DeepFilterNet 3 model weights during Docker build stage.
# This bakes the model weights (50MB) into the Docker image, preventing startup downloads
# and ensuring offline readiness. Placed before copying app source code for layer caching.
RUN python -c "import sys, types, torchaudio; b = types.ModuleType('torchaudio.backend'); sys.modules['torchaudio.backend'] = b; torchaudio.backend = b; m = types.ModuleType('torchaudio.backend.common'); m.AudioMetaData = getattr(torchaudio, 'AudioMetaData', None); sys.modules['torchaudio.backend.common'] = m; torchaudio.backend.common = m; import df.utils; df.utils.get_git_root = lambda: None; df.utils.get_commit_hash = lambda: None; from df.enhance import init_df; init_df()"

# Copy application source code
COPY app/ ./app/

# Expose port and run server
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

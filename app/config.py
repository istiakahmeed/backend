"""
Application configuration settings.
"""
import os
import tempfile
from pathlib import Path


# --- File Upload ---
MAX_FILE_SIZE_MB: int = 100
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS: set[str] = {".mp3", ".wav", ".m4a", ".aac", ".flac"}

ALLOWED_MIME_TYPES: set[str] = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/x-m4a",
    "audio/m4a",
    "audio/mp4",
    "audio/aac",
    "audio/flac",
    "audio/x-flac",
}

# --- Processing ---
DEEPFILTER_SAMPLE_RATE: int = 48000
PROCESSING_SAMPLE_RATE: int = 48000
TARGET_LUFS: float = -16.0

# Noise reduction aggressiveness (0.0 to 1.0)
NOISE_REDUCE_PROP_DECREASE: float = 0.85

# --- Temp Storage ---
TEMP_DIR: Path = Path(
    os.environ.get("TEMP_DIR", os.path.join(tempfile.gettempdir(), "noisecleaner"))
)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# How long to keep processed files before cleanup (seconds)
FILE_TTL_SECONDS: int = 3600  # 1 hour

# --- Server ---
_cors_env = os.environ.get("CORS_ORIGINS")
if _cors_env:
    CORS_ORIGINS: list[str] = [origin.strip() for origin in _cors_env.split(",") if origin.strip()]
else:
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "https://frontend-chi-six-45.vercel.app/"
    ]

# --- FFmpeg ---
FFMPEG_PATH: str = os.environ.get("FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH: str = os.environ.get("FFPROBE_PATH", "ffprobe")

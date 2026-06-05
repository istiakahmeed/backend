"""
Application configuration settings.
Enhanced pipeline configuration for multi-stage audio processing.

Voice-preservation-first philosophy:
  Default "balanced" mode prioritizes natural speech quality
  over aggressive noise removal. A small amount of residual
  background noise is preferable to robotic/metallic artifacts.
"""
import os
import tempfile
from pathlib import Path
from enum import Enum


# ============================================================
# Quality Modes
# ============================================================
class QualityMode(str, Enum):
    """
    Processing quality modes — controls aggressiveness across all pipeline stages.
    BALANCED is the default: natural voice with moderate noise removal.
    SOCIAL_MEDIA is optimized for mobile/social media publishing.
    """
    LIGHT = "light"          # Minimal processing, maximum voice preservation
    BALANCED = "balanced"    # Default — natural voice, moderate noise removal
    STRONG = "strong"        # More aggressive, slight voice impact possible
    MAXIMUM = "maximum"      # Maximum suppression, highest artifact risk
    SOCIAL_MEDIA = "social_media" # Optimized voice clarity & mobile compression target (-14 LUFS)
    PURE_VOICE = "pure_voice" # 100% original voice preservation, pure background noise removal

DEFAULT_QUALITY_MODE = QualityMode.BALANCED


# Per-mode parameter presets
# Calibrated for effective noise removal while preserving natural voice.
# DeepFilterNet is voice-aware by design, so moderate attenuation is safe.
# The robotic artifacts came from STACKING too many aggressive stages,
# not from any single stage being moderate.
QUALITY_MODE_PARAMS: dict[str, dict] = {
    QualityMode.PURE_VOICE: {
        "deepfilter_atten_db": 22,       # Moderate AI denoising (removes background noise)
        "speech_prop_decrease": 0.0,     # NO spectral gating on speech (preserves original voice harmonics)
        "nonspeech_prop_decrease": 0.70, # Moderate gating on silences
        "rnnoise_enabled": False,        # Skip RNNoise entirely to avoid artificial coloration
        "rnnoise_mix": 0.0,
        "adaptive_scale": 0.0,           # Skip adaptive suppression on speech frequencies
        "preprocess_hum_gain": 0,        # Bypass hum filter
        "eq_warmth_db": 0.0,             # Bypass warmth EQ
        "eq_boxiness_db": 0.0,           # Bypass mud cut
        "eq_presence_db": 0.0,           # Bypass presence boost
        "eq_sheen_db": 0.0,              # Bypass air sheen
        "comp_threshold": 0,
        "comp_ratio": 1.0,               # 1.0 ratio = no dynamic range compression
        "target_lufs": -16.0,            # Podcast loudness standard
    },
    QualityMode.LIGHT: {
        "deepfilter_atten_db": 15,       # Gentle AI denoising
        "speech_prop_decrease": 0.20,    # Light reduction on speech
        "nonspeech_prop_decrease": 0.60, # Moderate on silence
        "rnnoise_enabled": False,        # Skip RNNoise — DeepFilter alone is enough
        "rnnoise_mix": 0.0,
        "adaptive_scale": 0.3,          # Gentle adaptive suppression
        "preprocess_hum_gain": -8,
        "eq_warmth_db": 0.5,
        "eq_boxiness_db": -0.5,          # Subtle mud cut
        "eq_presence_db": 1.2,           # Gentle clarity boost
        "eq_sheen_db": 0.5,              # Gentle sheen
        "comp_threshold": -14,           # Light compression
        "comp_ratio": 1.5,
        "target_lufs": -16.0,            # Podcast loudness standard
    },
    QualityMode.BALANCED: {
        "deepfilter_atten_db": 28,       # Effective noise removal (DeepFilter is voice-safe)
        "speech_prop_decrease": 0.45,    # Smooth spectral gating on speech - less pumping
        "nonspeech_prop_decrease": 0.80, # Smooth spectral gating on silence - less pumping
        "rnnoise_enabled": True,
        "rnnoise_mix": 0.80,            # Let RNNoise contribute
        "adaptive_scale": 0.5,          # Moderate adaptive suppression
        "preprocess_hum_gain": -12,
        "eq_warmth_db": 1.0,
        "eq_boxiness_db": -1.5,          # Mid-range mud cleanup
        "eq_presence_db": 1.8,           # Vocal clarity boost
        "eq_sheen_db": 1.2,              # Professional sheen (8kHz)
        "comp_threshold": -20,
        "comp_ratio": 2.0,
        "target_lufs": -16.0,            # Podcast loudness standard
    },
    QualityMode.STRONG: {
        "deepfilter_atten_db": 38,       # Firm AI denoising
        "speech_prop_decrease": 0.60,    # Firm on speech
        "nonspeech_prop_decrease": 0.90, # Firm on silence
        "rnnoise_enabled": True,
        "rnnoise_mix": 0.90,            # 90% denoised
        "adaptive_scale": 0.75,
        "preprocess_hum_gain": -15,
        "eq_warmth_db": 1.0,
        "eq_boxiness_db": -1.5,
        "eq_presence_db": 2.0,
        "eq_sheen_db": 1.5,
        "comp_threshold": -22,
        "comp_ratio": 2.5,
        "target_lufs": -16.0,            # Podcast loudness standard
    },
    QualityMode.MAXIMUM: {
        "deepfilter_atten_db": 60,       # Heavy AI denoising
        "speech_prop_decrease": 0.65,    # Aggressive on speech
        "nonspeech_prop_decrease": 0.97, # Near-total on silence
        "rnnoise_enabled": True,
        "rnnoise_mix": 0.95,            # Nearly full denoised
        "adaptive_scale": 1.0,
        "preprocess_hum_gain": -18,
        "eq_warmth_db": 1.2,
        "eq_boxiness_db": -2.0,
        "eq_presence_db": 2.5,
        "eq_sheen_db": 1.8,
        "comp_threshold": -22,
        "comp_ratio": 3.0,
        "target_lufs": -16.0,            # Podcast loudness standard
    },
    QualityMode.SOCIAL_MEDIA: {
        "deepfilter_atten_db": 25,       # Calibrated to prevent robotic speech artifacts
        "speech_prop_decrease": 0.35,    # Very gentle VAD spectral gating for voice restoration
        "nonspeech_prop_decrease": 0.82, # Keep silences clean
        "rnnoise_enabled": True,
        "rnnoise_mix": 0.75,            # RNN high frequency cleanup
        "adaptive_scale": 0.5,
        "preprocess_hum_gain": -12,
        # Mobile-optimized parametric EQ & dynamic mastering
        "eq_warmth_db": 1.5,            # Warm vocal body (200Hz boost)
        "eq_boxiness_db": -2.0,          # Cut boxy room resonances (600Hz cut)
        "eq_presence_db": 2.8,          # Boost voice presence (3kHz) for clear mobile speaker playback
        "eq_sheen_db": 2.2,             # Air sheen (8kHz boost)
        "comp_threshold": -16,          # Stronger level leveling for social feeds
        "comp_ratio": 2.5,
        "target_lufs": -14.0,           # Loudness standard for Social Media (YouTube/TikTok)
    },
}


def get_mode_params(mode: str) -> dict:
    """Get parameter preset for a quality mode. Defaults to BALANCED."""
    return QUALITY_MODE_PARAMS.get(mode, QUALITY_MODE_PARAMS[QualityMode.BALANCED])


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

# Legacy fallback (used if no quality mode specified)
NOISE_REDUCE_PROP_DECREASE: float = 0.25

# --- Voice Preservation Band ---
VOICE_FREQ_LOW: int = 80       # Hz — lowest male fundamental
VOICE_FREQ_HIGH: int = 8000    # Hz — upper harmonics / sibilance

# --- Voice Activity Detection ---
# Tuned for generous speech detection to avoid cutting speech
VAD_AGGRESSIVENESS: int = 2               # Balanced WebRTC VAD mode (was 1)
VAD_MIN_SPEECH_MS: int = 150              # Shorter min speech to catch syllables (was 250)
VAD_MIN_SILENCE_MS: int = 300             # Longer min silence to avoid choppy (was 150)
VAD_ENERGY_THRESHOLD: float = 0.010       # Lower threshold, detect more speech (was 0.015)
VAD_FRAME_DURATION_MS: int = 20
VAD_CROSSFADE_MS: int = 50               # Wider crossfade for smooth transitions (was 30)

# Legacy per-region settings (overridden by quality mode)
NOISE_REDUCE_PROP_DECREASE_SPEECH: float = 0.25
NOISE_REDUCE_PROP_DECREASE_NONSPEECH: float = 0.70

# Legacy DeepFilter settings (overridden by quality mode)
DEEPFILTER_ATTEN_LIM_DEFAULT: int = 18
DEEPFILTER_ATTEN_LIM_MAX_MODE: int = 60

# --- Quality Scoring ---
QUALITY_SCORE_ENABLED: bool = True

# --- RNNoise ---
RNNOISE_ENABLED: bool = True

# --- Temp Storage ---
TEMP_DIR: Path = Path(
    os.environ.get("TEMP_DIR", os.path.join(tempfile.gettempdir(), "noisecleaner"))
)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

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
        "https://frontend-chi-six-45.vercel.app"
    ]

# --- FFmpeg ---
FFMPEG_PATH: str = os.environ.get("FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH: str = os.environ.get("FFPROBE_PATH", "ffprobe")

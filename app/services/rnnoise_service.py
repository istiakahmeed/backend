"""
RNNoise speech denoising service.
Uses FFmpeg's built-in arnndn filter for RNN-based noise suppression.

Voice preservation:
  - Uses dry/wet mix parameter to blend denoised with original
  - Balanced mode uses 50% mix (not 100%) to preserve vocal texture
  - Light mode skips RNNoise entirely
  - Graceful fallback if FFmpeg lacks arnndn support
"""
import logging
import shutil
import subprocess

from app.config import (
    FFMPEG_PATH,
    PROCESSING_SAMPLE_RATE,
    RNNOISE_ENABLED,
    QualityMode,
    get_mode_params,
)

logger = logging.getLogger(__name__)

_rnnoise_available: bool | None = None


def check_rnnoise_support() -> bool:
    """Check if the installed FFmpeg has arnndn (RNNoise) filter support."""
    global _rnnoise_available

    if _rnnoise_available is not None:
        return _rnnoise_available

    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-filters"],
            capture_output=True, text=True, timeout=10,
        )
        _rnnoise_available = "arnndn" in result.stdout
        if _rnnoise_available:
            logger.info("FFmpeg arnndn (RNNoise) filter: available")
        else:
            logger.info("FFmpeg arnndn (RNNoise) filter: not available")
    except Exception as e:
        logger.warning(f"Could not check FFmpeg RNNoise support: {e}")
        _rnnoise_available = False

    return _rnnoise_available


def apply_rnnoise(
    input_path: str,
    output_path: str,
    quality_mode: str = "balanced",
) -> str:
    """
    Apply RNNoise denoising via FFmpeg's arnndn filter with dry/wet mix.

    The mix parameter controls how much of the denoised signal to use:
      - mix=0.0: fully original (no denoising)
      - mix=0.5: 50% denoised, 50% original (balanced default)
      - mix=1.0: fully denoised (maximum)

    Args:
        input_path: Path to input WAV file (48kHz mono)
        output_path: Path to write denoised output
        quality_mode: Quality mode string

    Returns:
        Path to the output file
    """
    params = get_mode_params(quality_mode)

    # Check if RNNoise should be used for this mode
    if not params.get("rnnoise_enabled", True):
        logger.info(f"RNNoise disabled for {quality_mode} mode, skipping")
        shutil.copy2(input_path, output_path)
        return output_path

    if not RNNOISE_ENABLED:
        logger.info("RNNoise disabled in config, skipping")
        shutil.copy2(input_path, output_path)
        return output_path

    if not check_rnnoise_support():
        logger.info("RNNoise not available in FFmpeg, skipping stage")
        shutil.copy2(input_path, output_path)
        return output_path

    mix_value = params.get("rnnoise_mix", 0.5)
    logger.info(f"Applying RNNoise: mode={quality_mode}, mix={mix_value}")

    # Build arnndn filter with model path
    from pathlib import Path
    model_path = Path(__file__).parent / "cb.rnnn"
    
    if model_path.exists():
        filter_str = f"arnndn=model={model_path}:mix={mix_value}" if mix_value < 1.0 else f"arnndn=model={model_path}"
    else:
        logger.warning("cb.rnnn model file not found, falling back to default arnndn")
        filter_str = f"arnndn=mix={mix_value}" if mix_value < 1.0 else "arnndn"

    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", input_path,
        "-af", filter_str,
        "-ar", str(PROCESSING_SAMPLE_RATE),
        "-ac", "1",
        "-sample_fmt", "s16",
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=180,
        )
        logger.info(f"RNNoise denoising complete (mix={mix_value}): {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        logger.warning(f"RNNoise processing failed: {e.stderr}")
        # Try without mix parameter as fallback (still including the model if available)
        try:
            fallback_filter = f"arnndn=model={model_path}" if model_path.exists() else "arnndn"
            cmd_fallback = [
                FFMPEG_PATH, "-y",
                "-i", input_path,
                "-af", fallback_filter,
                "-ar", str(PROCESSING_SAMPLE_RATE),
                "-ac", "1", "-sample_fmt", "s16",
                output_path,
            ]
            subprocess.run(
                cmd_fallback, capture_output=True, text=True, check=True, timeout=180,
            )
            logger.info("RNNoise fallback (no mix param) succeeded")
            return output_path
        except Exception as fallback_err:
            logger.warning(f"RNNoise fallback failed: {fallback_err}")
            pass

        logger.info("RNNoise failed, copying original")
        shutil.copy2(input_path, output_path)
        return output_path

    except subprocess.TimeoutExpired:
        logger.warning("RNNoise timed out, skipping")
        shutil.copy2(input_path, output_path)
        return output_path


def is_available() -> bool:
    """Check if RNNoise is enabled and FFmpeg supports it."""
    return RNNOISE_ENABLED and check_rnnoise_support()


def get_info() -> dict:
    """Return RNNoise status for health checks."""
    return {
        "enabled": RNNOISE_ENABLED,
        "ffmpeg_support": check_rnnoise_support() if RNNOISE_ENABLED else None,
    }

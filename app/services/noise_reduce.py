"""
Noise reduction service using spectral gating (noisereduce library).
Applied as a secondary pass after DeepFilterNet for residual noise cleanup.
"""
import logging
import numpy as np
import soundfile as sf

from app.config import NOISE_REDUCE_PROP_DECREASE, PROCESSING_SAMPLE_RATE

logger = logging.getLogger(__name__)


def reduce_residual_noise(input_path: str, output_path: str) -> str:
    """
    Apply spectral gating noise reduction to clean up residual noise
    left after DeepFilterNet processing.

    Uses non-stationary mode for adaptive noise threshold.

    Args:
        input_path: Path to input WAV file
        output_path: Path to write cleaned WAV file

    Returns:
        Path to the cleaned audio file
    """
    import noisereduce as nr

    logger.info(f"Spectral gating noise reduction: {input_path}")

    try:
        # Load audio
        audio_data, sr = sf.read(input_path, dtype="float32")

        logger.info(
            f"Audio loaded for noise reduction: "
            f"shape={audio_data.shape}, sr={sr}, "
            f"duration={len(audio_data) / sr:.1f}s"
        )

        # Apply non-stationary noise reduction
        # This adapts the noise threshold over time, better for varied noise
        reduced = nr.reduce_noise(
            y=audio_data,
            sr=sr,
            stationary=False,
            prop_decrease=NOISE_REDUCE_PROP_DECREASE,
            n_fft=2048,
            hop_length=512,
            n_std_thresh_stationary=1.5,
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50,
        )

        # Ensure no clipping
        reduced = np.clip(reduced, -1.0, 1.0)

        # Write output
        sf.write(output_path, reduced, sr, subtype="PCM_16")

        logger.info(f"Spectral gating complete: {output_path}")
        return output_path

    except ImportError:
        logger.error("noisereduce not installed. Install with: pip install noisereduce")
        raise
    except Exception as e:
        logger.error(f"Noise reduction failed: {e}")
        raise RuntimeError(f"Spectral gating noise reduction failed: {e}")

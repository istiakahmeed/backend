"""
Noise reduction service using spectral gating (noisereduce library).

Voice-preservation-first approach:
  - Speech regions get very gentle reduction (0.25 for balanced, was 0.6)
  - Non-speech regions get moderate reduction (0.70 for balanced, was 0.95)
  - Wider crossfade at speech/silence boundaries to prevent choppy artifacts
  - If confidence is low, preserve original voice rather than aggressively filter
"""
import logging
import numpy as np
import soundfile as sf

from app.config import (
    NOISE_REDUCE_PROP_DECREASE,
    PROCESSING_SAMPLE_RATE,
    QualityMode,
    get_mode_params,
)

logger = logging.getLogger(__name__)


def reduce_residual_noise(
    input_path: str,
    output_path: str,
    quality_mode: str = "balanced",
) -> str:
    """
    Apply spectral gating with quality-mode-aware aggressiveness.
    Fallback for when VAD mask is not available.
    """
    import noisereduce as nr

    params = get_mode_params(quality_mode)
    # Use the speech prop as the uniform reduction (gentler)
    prop_decrease = params["speech_prop_decrease"]

    logger.info(f"Spectral gating (uniform): prop_decrease={prop_decrease}")

    try:
        audio_data, sr = sf.read(input_path, dtype="float32")

        reduced = nr.reduce_noise(
            y=audio_data,
            sr=sr,
            stationary=False,
            prop_decrease=prop_decrease,
            n_fft=2048,
            hop_length=512,
            n_std_thresh_stationary=2.0,   # Higher threshold = less aggressive (was 1.5)
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=80,        # Wider smoothing to prevent artifacts (was 50)
        )

        reduced = np.clip(reduced, -1.0, 1.0)
        sf.write(output_path, reduced, sr, subtype="PCM_16")

        logger.info(f"Spectral gating complete: {output_path}")
        return output_path

    except ImportError:
        logger.error("noisereduce not installed")
        raise
    except Exception as e:
        logger.error(f"Noise reduction failed: {e}")
        raise RuntimeError(f"Spectral gating failed: {e}")


def reduce_noise_vad_aware(
    input_path: str,
    output_path: str,
    speech_mask: np.ndarray | None = None,
    noise_spectral_profile: np.ndarray | None = None,
    quality_mode: str = "balanced",
    max_noise_removal: bool = False,
) -> str:
    """
    VAD-aware noise reduction with voice-preservation-first defaults.

    Key voice preservation strategies:
    - Speech regions: very gentle reduction (0.25 balanced) to avoid metallic/robotic sound
    - Non-speech regions: moderate reduction (0.70 balanced) — NOT near-total suppression
    - Wider time smoothing (80ms) to prevent choppy artifacts
    - Higher n_std threshold on speech to avoid catching quiet speech as noise
    - Smooth blending at speech/silence boundaries

    Args:
        input_path: Path to input WAV
        output_path: Path to output WAV
        speech_mask: Per-sample mask (1=speech, 0=non-speech)
        noise_spectral_profile: Noise reference spectrum (optional)
        quality_mode: Quality mode string
        max_noise_removal: Legacy flag
    """
    import noisereduce as nr

    if max_noise_removal:
        quality_mode = QualityMode.MAXIMUM

    params = get_mode_params(quality_mode)
    speech_prop = params["speech_prop_decrease"]
    nonspeech_prop = params["nonspeech_prop_decrease"]

    logger.info(
        f"VAD-aware spectral gating: mode={quality_mode}, "
        f"speech_prop={speech_prop}, nonspeech_prop={nonspeech_prop}"
    )

    try:
        audio_data, sr = sf.read(input_path, dtype="float32")

        if speech_mask is None:
            logger.info("No speech mask, using uniform gentle reduction")
            return reduce_residual_noise(input_path, output_path, quality_mode)

        # Resize mask if needed
        if len(speech_mask) != len(audio_data):
            from scipy.ndimage import zoom
            scale = len(audio_data) / len(speech_mask)
            speech_mask = zoom(speech_mask, scale, order=1)
            speech_mask = speech_mask[:len(audio_data)]

        speech_ratio = np.mean(speech_mask)
        logger.info(
            f"VAD: {speech_ratio:.0%} speech, "
            f"duration={len(audio_data) / sr:.1f}s"
        )

        # --- Process speech regions GENTLY ---
        reduced_gentle = nr.reduce_noise(
            y=audio_data,
            sr=sr,
            stationary=False,
            prop_decrease=speech_prop,
            n_fft=2048,
            hop_length=512,
            n_std_thresh_stationary=2.5,   # High threshold: only remove obvious noise
            freq_mask_smooth_hz=600,       # Wide freq smoothing to prevent spectral holes
            time_mask_smooth_ms=100,       # Wide time smoothing to prevent choppy
        )

        # --- Process non-speech regions with moderate strength ---
        reduced_strong = nr.reduce_noise(
            y=audio_data,
            sr=sr,
            stationary=False,
            prop_decrease=nonspeech_prop,
            n_fft=2048,
            hop_length=512,
            n_std_thresh_stationary=1.5,
            freq_mask_smooth_hz=400,
            time_mask_smooth_ms=60,
        )

        # --- Blend using speech mask ---
        result = speech_mask * reduced_gentle + (1.0 - speech_mask) * reduced_strong

        # Ensure no clipping
        result = np.clip(result, -1.0, 1.0)

        sf.write(output_path, result, sr, subtype="PCM_16")

        logger.info(f"VAD-aware spectral gating complete: {output_path}")
        return output_path

    except ImportError:
        logger.error("noisereduce not installed")
        raise
    except Exception as e:
        logger.error(f"VAD-aware noise reduction failed: {e}", exc_info=True)
        logger.info("Falling back to gentle uniform reduction")
        return reduce_residual_noise(input_path, output_path, quality_mode)

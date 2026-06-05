"""
DeepFilterNet service for AI-powered noise reduction.
Wraps the DeepFilterNet 3 model for speech enhancement.

Voice-preservation tuning:
  - Default attenuation lowered to 18 dB (from 40) to prevent metallic artifacts
  - Quality mode controls aggressiveness
  - Post-enhancement dry/wet blending preserves vocal texture
"""
import logging
import numpy as np
import torch

from app.config import (
    DEEPFILTER_ATTEN_LIM_DEFAULT,
    DEEPFILTER_ATTEN_LIM_MAX_MODE,
    QualityMode,
    get_mode_params,
)

logger = logging.getLogger(__name__)

# Module-level model cache
_model = None
_df_state = None
_device = None


def init_model() -> bool:
    """
    Initialize DeepFilterNet model at startup.
    Caches the model in memory to avoid reloading on each request.
    """
    global _model, _df_state, _device

    if _model is not None:
        logger.info("DeepFilterNet model already loaded, skipping init")
        return True

    logger.info("Loading DeepFilterNet 3 model...")

    try:
        from df.enhance import init_df

        if torch.backends.mps.is_available():
            _device = "mps"
        elif torch.cuda.is_available():
            _device = "cuda"
        else:
            _device = "cpu"

        logger.info(f"Using device: {_device}")

        _model, _df_state, _ = init_df()

        logger.info(
            f"DeepFilterNet model loaded successfully "
            f"(sr={_df_state.sr()}, device={_device})"
        )
        return True

    except ImportError:
        logger.warning(
            "DeepFilterNet not installed. Install with: pip install deepfilternet"
        )
        return False
    except Exception as e:
        logger.warning(
            f"Failed to load DeepFilterNet model: {e}. "
            f"Will use alternative noise reduction methods."
        )
        return False


def is_model_available() -> bool:
    """Check if DeepFilterNet model is loaded and ready."""
    return _model is not None and _df_state is not None


def enhance_audio(input_path: str, output_path: str) -> str:
    """Backward-compatible wrapper using default attenuation."""
    return enhance_audio_with_params(
        input_path, output_path,
        atten_lim_db=DEEPFILTER_ATTEN_LIM_DEFAULT,
    )


def enhance_audio_with_params(
    input_path: str,
    output_path: str,
    atten_lim_db: int = DEEPFILTER_ATTEN_LIM_DEFAULT,
    quality_mode: str = "balanced",
    max_noise_removal: bool = False,
) -> str:
    """
    Enhance audio using DeepFilterNet 3 with quality-mode-aware parameters.

    Key voice preservation strategies:
    - Use moderate attenuation (18 dB balanced, not 40+)
    - Blend enhanced audio with original (dry/wet mix) to preserve vocal texture
    - Clamp output to prevent distortion

    Args:
        input_path: Path to input WAV file (48kHz, mono)
        output_path: Path to write enhanced WAV file
        atten_lim_db: Attenuation limit in dB
        quality_mode: Quality mode string (light/balanced/strong/maximum)
        max_noise_removal: Legacy flag — if True, uses maximum mode
    """
    import shutil
    global _model, _df_state

    if _model is None or _df_state is None:
        logger.warning(
            "DeepFilterNet model not available. Skipping AI enhancement."
        )
        shutil.copy2(input_path, output_path)
        return output_path

    from df.enhance import enhance, load_audio, save_audio

    # Resolve quality mode parameters
    if max_noise_removal:
        quality_mode = QualityMode.MAXIMUM
    params = get_mode_params(quality_mode)
    atten_lim_db = params["deepfilter_atten_db"]

    logger.info(
        f"DeepFilterNet processing: mode={quality_mode}, "
        f"atten_lim={atten_lim_db}dB"
    )

    try:
        # Load audio at the model's expected sample rate
        audio, sr_info = load_audio(input_path, sr=_df_state.sr())

        sr_value = getattr(sr_info, "sample_rate", sr_info)
        if not isinstance(sr_value, (int, float)):
            sr_value = _df_state.sr()

        logger.info(
            f"Audio loaded: shape={audio.shape}, sr={sr_value}, "
            f"duration={audio.shape[-1] / sr_value:.1f}s"
        )

        # Run enhancement with attenuation limit
        try:
            enhanced_audio = enhance(
                _model, _df_state, audio,
                atten_lim_db=atten_lim_db,
            )
        except TypeError:
            logger.info("Falling back to enhance() without atten_lim_db parameter")
            enhanced_audio = enhance(_model, _df_state, audio)

        # ---- DRY/WET BLENDING for voice preservation ----
        # Blend enhanced with original to retain vocal texture.
        # Calibrated to remove noise while keeping vocal naturalness.
        blend_ratios = {
            QualityMode.LIGHT: 0.80,     # 80% enhanced, 20% original
            QualityMode.BALANCED: 0.93,   # 93% enhanced, 7% original (sweet spot)
            QualityMode.STRONG: 0.98,     # 98% enhanced, 2% original
            QualityMode.MAXIMUM: 1.0,     # 100% enhanced
        }
        wet_ratio = blend_ratios.get(quality_mode, 0.93)

        if wet_ratio < 1.0:
            if isinstance(enhanced_audio, torch.Tensor):
                blended = wet_ratio * enhanced_audio + (1.0 - wet_ratio) * audio
                enhanced_audio = torch.clamp(blended, -1.0, 1.0)
            else:
                blended = wet_ratio * enhanced_audio + (1.0 - wet_ratio) * audio.numpy() if isinstance(audio, torch.Tensor) else audio
                enhanced_audio = np.clip(blended, -1.0, 1.0)

            logger.info(f"Dry/wet blend: {wet_ratio:.0%} enhanced, {1-wet_ratio:.0%} original")
        else:
            if isinstance(enhanced_audio, torch.Tensor):
                enhanced_audio = torch.clamp(enhanced_audio, -1.0, 1.0)
            else:
                enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)

        save_audio(output_path, enhanced_audio, sr=_df_state.sr())

        logger.info(f"DeepFilterNet enhancement complete: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"DeepFilterNet processing failed: {e}")
        logger.info("Falling back to copying audio without AI enhancement")
        shutil.copy2(input_path, output_path)
        return output_path


def get_model_info() -> dict:
    """Return info about the loaded model for health checks."""
    return {
        "loaded": _model is not None,
        "device": str(_device) if _device else None,
        "sample_rate": _df_state.sr() if _df_state else None,
        "default_atten_db": DEEPFILTER_ATTEN_LIM_DEFAULT,
    }

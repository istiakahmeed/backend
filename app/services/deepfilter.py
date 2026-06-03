"""
DeepFilterNet service for AI-powered noise reduction.
Wraps the DeepFilterNet 3 model for speech enhancement.
"""
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Module-level model cache
_model = None
_df_state = None
_device = None


def init_model() -> None:
    """
    Initialize DeepFilterNet model at startup.
    Caches the model in memory to avoid reloading on each request.
    """
    global _model, _df_state, _device

    if _model is not None:
        logger.info("DeepFilterNet model already loaded, skipping init")
        return

    logger.info("Loading DeepFilterNet 3 model...")

    try:
        from df.enhance import init_df

        # Select device: MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available():
            _device = "mps"
        elif torch.cuda.is_available():
            _device = "cuda"
        else:
            _device = "cpu"

        logger.info(f"Using device: {_device}")

        # init_df returns (model, df_state, _)
        # For CPU/MPS we pass post_filter=True for better quality
        _model, _df_state, _ = init_df()

        logger.info(
            f"DeepFilterNet model loaded successfully "
            f"(sr={_df_state.sr()}, device={_device})"
        )

    except ImportError:
        logger.error(
            "DeepFilterNet not installed. Install with: pip install deepfilternet"
        )
        raise
    except Exception as e:
        logger.error(f"Failed to load DeepFilterNet model: {e}")
        raise


def enhance_audio(input_path: str, output_path: str) -> str:
    """
    Enhance audio using DeepFilterNet 3.
    Removes background noise while preserving voice quality.

    Args:
        input_path: Path to input WAV file (48kHz, mono, 16-bit)
        output_path: Path to write enhanced WAV file

    Returns:
        Path to the enhanced audio file
    """
    global _model, _df_state

    if _model is None or _df_state is None:
        raise RuntimeError("DeepFilterNet model not initialized. Call init_model() first.")

    from df.enhance import enhance, load_audio, save_audio

    logger.info(f"DeepFilterNet processing: {input_path}")

    try:
        # Load audio at the model's expected sample rate
        audio, sr_info = load_audio(input_path, sr=_df_state.sr())

        # sr_info may be an AudioMetaData object in newer torchaudio versions
        sr_value = getattr(sr_info, "sample_rate", sr_info)
        if not isinstance(sr_value, (int, float)):
            sr_value = _df_state.sr()

        logger.info(
            f"Audio loaded: shape={audio.shape}, sr={sr_value}, "
            f"duration={audio.shape[-1] / sr_value:.1f}s"
        )

        # Run enhancement
        enhanced_audio = enhance(_model, _df_state, audio)

        # Clamp to prevent clipping
        if isinstance(enhanced_audio, torch.Tensor):
            enhanced_audio = torch.clamp(enhanced_audio, -1.0, 1.0)
        else:
            enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)

        # Save enhanced audio
        save_audio(output_path, enhanced_audio, sr=_df_state.sr())

        logger.info(f"DeepFilterNet enhancement complete: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"DeepFilterNet processing failed: {e}")
        raise RuntimeError(f"AI noise reduction failed: {e}")


def get_model_info() -> dict:
    """Return info about the loaded model for health checks."""
    return {
        "loaded": _model is not None,
        "device": str(_device) if _device else None,
        "sample_rate": _df_state.sr() if _df_state else None,
    }

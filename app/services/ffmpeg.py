"""
FFmpeg service for audio format conversion, normalization, and enhancement.

Voice-preservation philosophy:
  - Preprocessing: very gentle de-hum and minimal compression
  - Final enhancement: subtle EQ, light compression, NO aggressive processing
  - All filter strengths controlled by quality mode
"""
import subprocess
import logging
import json
from pathlib import Path

from app.config import (
    FFMPEG_PATH, FFPROBE_PATH, PROCESSING_SAMPLE_RATE, TARGET_LUFS,
    QualityMode, get_mode_params,
)

logger = logging.getLogger(__name__)


def get_audio_info(input_path: str) -> dict:
    """Get audio file metadata using ffprobe."""
    cmd = [
        FFPROBE_PATH,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        input_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe failed: {e.stderr}")
        raise RuntimeError(f"Failed to analyze audio file: {e.stderr}")


def convert_to_processing_format(input_path: str, output_path: str) -> str:
    """Convert any supported audio file to mono 48kHz WAV for processing."""
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", input_path,
        "-ar", str(PROCESSING_SAMPLE_RATE),
        "-ac", "1",
        "-sample_fmt", "s16",
        "-f", "wav",
        output_path,
    ]

    logger.info(f"Converting to processing format: {input_path} → {output_path}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=120
        )
        logger.info("Conversion complete")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr}")
        raise RuntimeError(f"Audio conversion failed: {e.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Audio conversion timed out (>120s)")


def preprocess_audio(
    input_path: str,
    output_path: str,
    quality_mode: str = "balanced",
) -> str:
    """
    Gentle audio preprocessing — only removes obvious problems.

    Voice preservation: keep preprocessing MINIMAL to avoid
    cascading artifacts through the pipeline.

    - Highpass at 55 Hz (only remove deep subsonic rumble)
    - Very gentle hum notch filters (scaled by quality mode)
    - NO compression in preprocessing (was causing pumping artifacts)

    Args:
        input_path: Path to input WAV
        output_path: Path to write preprocessed WAV
        quality_mode: Quality mode string
    """
    params = get_mode_params(quality_mode)
    hum_gain = params.get("preprocess_hum_gain", -10)

    filters = [
        # Remove DC offset
        "dcshift=0",
    ]
    if quality_mode != "pure_voice":
        # Gentle highpass — only remove deep sub-bass rumble
        filters.append("highpass=f=55:p=1")

    if hum_gain != 0:
        # Very gentle hum removal at 50 Hz and 60 Hz
        # Using narrow notch (high Q) to avoid affecting voice
        filters.append(f"equalizer=f=50:t=q:w=8:g={hum_gain}")
        filters.append(f"equalizer=f=60:t=q:w=8:g={hum_gain}")
        # Only remove strong harmonics, not gentle ones
        filters.append(f"equalizer=f=100:t=q:w=8:g={hum_gain * 0.5:.0f}")
        filters.append(f"equalizer=f=120:t=q:w=8:g={hum_gain * 0.5:.0f}")

    filter_chain = ",".join(filters)

    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", input_path,
        "-af", filter_chain,
        "-ar", str(PROCESSING_SAMPLE_RATE),
        "-ac", "1",
        "-sample_fmt", "s16",
        output_path,
    ]

    logger.info(f"Gentle preprocessing (hum_gain={hum_gain}dB): {input_path}")

    try:
        subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=120
        )
        logger.info("Preprocessing complete")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.warning(f"Preprocessing failed: {e.stderr}")
        import shutil
        shutil.copy2(input_path, output_path)
        return output_path
    except subprocess.TimeoutExpired:
        logger.warning("Preprocessing timed out")
        import shutil
        shutil.copy2(input_path, output_path)
        return output_path


def apply_final_enhancement(
    input_path: str,
    output_path: str,
    quality_mode: str = "balanced",
) -> str:
    """
    Final subtle voice enhancement for natural-sounding, professional output.
    
    Implementing:
    - Stage 6: Parametric EQ (warmth at 200Hz, boxiness cut at 600Hz)
    - Stage 7: Dynamic Compression (slow attack leveling)
    - Stage 9: Presence & Sheen (vocal clarity boost at 3kHz, high shelf sheen at 8kHz)
    - Stage 10: Final Mastering (safety limiting at -1.5dB true peak)
    """
    params = get_mode_params(quality_mode)
    warmth_db = params.get("eq_warmth_db", 1.0)
    boxiness_db = params.get("eq_boxiness_db", -1.5)
    presence_db = params.get("eq_presence_db", 1.8)
    sheen_db = params.get("eq_sheen_db", 1.2)
    comp_threshold = params.get("comp_threshold", -20)
    comp_ratio = params.get("comp_ratio", 2.0)

    filters = []

    # Subsonic cleanup - bypass in pure_voice mode
    if quality_mode != "pure_voice":
        filters.append("highpass=f=65:p=1")

    # EQ stages (only add if gain is non-zero)
    if warmth_db != 0.0:
        filters.append(f"equalizer=f=200:t=q:w=1.0:g={warmth_db}")
    if boxiness_db != 0.0:
        filters.append(f"equalizer=f=600:t=q:w=0.8:g={boxiness_db}")
    if presence_db != 0.0:
        filters.append(f"equalizer=f=3000:t=q:w=1.0:g={presence_db}")
    if sheen_db != 0.0:
        filters.append(f"equalizer=f=8000:t=q:w=0.8:g={sheen_db}")

    # Dynamic Compression (only add if ratio is greater than 1.0)
    if comp_ratio > 1.0:
        filters.append(f"acompressor=threshold={comp_threshold}dB:ratio={comp_ratio}:attack=25:release=150:knee=8:makeup=1.5")

    # Safety Limiter: -1.5dB ceiling to prevent distortion
    filters.append("alimiter=limit=0.85:attack=5:release=50")

    filter_chain = ",".join(filters)

    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", input_path,
        "-af", filter_chain,
        "-ar", str(PROCESSING_SAMPLE_RATE),
        "-ac", "1",
        "-sample_fmt", "s16",
        output_path,
    ]

    logger.info(
        f"Final enhancement: warmth={warmth_db}dB, boxiness={boxiness_db}dB, "
        f"presence={presence_db}dB, sheen={sheen_db}dB, "
        f"comp={comp_threshold}dB/{comp_ratio}:1"
    )

    try:
        subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=120
        )
        logger.info("Final enhancement complete")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.warning(f"Final enhancement failed: {e.stderr}")
        import shutil
        shutil.copy2(input_path, output_path)
        return output_path
    except subprocess.TimeoutExpired:
        logger.warning("Final enhancement timed out")
        import shutil
        shutil.copy2(input_path, output_path)
        return output_path


def normalize_audio(input_path: str, output_path: str, quality_mode: str = "balanced") -> str:
    """
    Normalize audio loudness using FFmpeg's loudnorm filter.
    Supports variable target LUFS based on quality mode presets (-14 LUFS social, -16 LUFS podcast).
    """
    params = get_mode_params(quality_mode)
    target_lufs = params.get("target_lufs", TARGET_LUFS)

    measure_cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", input_path,
        "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json",
        "-f", "null",
        "-",
    ]

    logger.info(f"Loudness normalization (target={target_lufs} LUFS) pass 1: measuring...")

    try:
        result = subprocess.run(
            measure_cmd, capture_output=True, text=True, check=True, timeout=120
        )
        stderr = result.stderr
        json_start = stderr.rfind("{")
        json_end = stderr.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            logger.warning("Could not parse loudnorm stats, falling back to single-pass")
            return _single_pass_normalize(input_path, output_path, target_lufs)

        stats = json.loads(stderr[json_start:json_end])

        normalize_cmd = [
            FFMPEG_PATH,
            "-y",
            "-i", input_path,
            "-af", (
                f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:"
                f"measured_I={stats['input_i']}:"
                f"measured_TP={stats['input_tp']}:"
                f"measured_LRA={stats['input_lra']}:"
                f"measured_thresh={stats['input_thresh']}:"
                f"offset={stats['target_offset']}:"
                f"linear=true:print_format=summary"
            ),
            "-ar", str(PROCESSING_SAMPLE_RATE),
            "-sample_fmt", "s16",
            output_path,
        ]

        logger.info("Loudness normalization pass 2: applying...")
        subprocess.run(
            normalize_cmd, capture_output=True, text=True, check=True, timeout=120
        )
        logger.info("Normalization complete")
        return output_path

    except subprocess.CalledProcessError as e:
        logger.warning(f"Two-pass normalization failed: {e.stderr}")
        return _single_pass_normalize(input_path, output_path, target_lufs)
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse loudnorm stats: {e}")
        return _single_pass_normalize(input_path, output_path, target_lufs)


def _single_pass_normalize(input_path: str, output_path: str, target_lufs: float = -16.0) -> str:
    """Fallback single-pass loudness normalization."""
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", input_path,
        "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
        "-ar", str(PROCESSING_SAMPLE_RATE),
        "-sample_fmt", "s16",
        output_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Single-pass normalization failed: {e.stderr}")
        raise RuntimeError(f"Audio normalization failed: {e.stderr}")


def export_to_format(
    input_path: str, output_path: str, output_format: str = "wav"
) -> str:
    """Export processed audio to WAV or MP3."""
    if output_format == "mp3":
        cmd = [
            FFMPEG_PATH,
            "-y",
            "-i", input_path,
            "-codec:a", "libmp3lame",
            "-qscale:a", "2",
            output_path,
        ]
    else:
        cmd = [
            FFMPEG_PATH,
            "-y",
            "-i", input_path,
            "-ar", str(PROCESSING_SAMPLE_RATE),
            "-sample_fmt", "s16",
            output_path,
        ]

    logger.info(f"Exporting to {output_format}: {output_path}")

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Export failed: {e.stderr}")
        raise RuntimeError(f"Audio export to {output_format} failed: {e.stderr}")


def check_rnnoise_support() -> bool:
    """Check if FFmpeg has arnndn (RNNoise) filter support."""
    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-filters"],
            capture_output=True, text=True, timeout=10,
        )
        return "arnndn" in result.stdout
    except Exception:
        return False

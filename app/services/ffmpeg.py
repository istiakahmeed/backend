"""
FFmpeg service for audio format conversion and normalization.
"""
import subprocess
import logging
from pathlib import Path

from app.config import FFMPEG_PATH, FFPROBE_PATH, PROCESSING_SAMPLE_RATE, TARGET_LUFS

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
        import json
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe failed: {e.stderr}")
        raise RuntimeError(f"Failed to analyze audio file: {e.stderr}")


def convert_to_processing_format(input_path: str, output_path: str) -> str:
    """
    Convert any supported audio file to mono 48kHz WAV for processing.
    DeepFilterNet requires 48kHz sample rate.
    """
    cmd = [
        FFMPEG_PATH,
        "-y",                        # Overwrite output
        "-i", input_path,            # Input file
        "-ar", str(PROCESSING_SAMPLE_RATE),  # 48kHz
        "-ac", "1",                  # Mono
        "-sample_fmt", "s16",        # 16-bit signed int
        "-f", "wav",                 # WAV format
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


def normalize_audio(input_path: str, output_path: str) -> str:
    """
    Normalize audio loudness using FFmpeg's loudnorm filter.
    Two-pass EBU R128 loudness normalization targeting -16 LUFS.
    """
    # Pass 1: Measure current loudness
    measure_cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", input_path,
        "-af", f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11:print_format=json",
        "-f", "null",
        "-",
    ]

    logger.info("Loudness normalization pass 1: measuring...")

    try:
        result = subprocess.run(
            measure_cmd, capture_output=True, text=True, check=True, timeout=120
        )
        # Parse loudnorm stats from stderr (FFmpeg outputs filter stats there)
        stderr = result.stderr
        # Find the JSON block in stderr
        import json
        json_start = stderr.rfind("{")
        json_end = stderr.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            logger.warning("Could not parse loudnorm stats, falling back to single-pass")
            return _single_pass_normalize(input_path, output_path)

        stats = json.loads(stderr[json_start:json_end])

        # Pass 2: Apply normalization with measured values
        normalize_cmd = [
            FFMPEG_PATH,
            "-y",
            "-i", input_path,
            "-af", (
                f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11:"
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
        logger.warning(f"Two-pass normalization failed, falling back: {e.stderr}")
        return _single_pass_normalize(input_path, output_path)
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse loudnorm stats: {e}")
        return _single_pass_normalize(input_path, output_path)


def _single_pass_normalize(input_path: str, output_path: str) -> str:
    """Fallback single-pass loudness normalization."""
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", input_path,
        "-af", f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11",
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
    """
    Export processed audio to the desired output format.
    Supports wav and mp3.
    """
    if output_format == "mp3":
        cmd = [
            FFMPEG_PATH,
            "-y",
            "-i", input_path,
            "-codec:a", "libmp3lame",
            "-qscale:a", "2",  # High quality VBR (~190kbps)
            output_path,
        ]
    else:  # wav (default)
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

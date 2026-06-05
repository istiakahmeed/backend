"""
Audio quality assessment and scoring system.

Computes quality metrics by comparing original and processed audio:
  - Input noise level (dB)
  - Output noise level (dB)
  - Noise reduction improvement (%)
  - Voice preservation score (0–100)

Provides a QualityReport included in job results.
"""
import logging
from dataclasses import dataclass

import numpy as np
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Quality assessment result."""
    input_noise_level_db: float = 0.0
    output_noise_level_db: float = 0.0
    noise_reduction_db: float = 0.0
    improvement_percentage: float = 0.0
    voice_preservation_score: float = 0.0   # 0–100
    input_snr_db: float = 0.0
    output_snr_db: float = 0.0
    overall_quality_score: float = 0.0      # 0–100

    def to_dict(self) -> dict:
        return {
            "input_noise_level_db": round(self.input_noise_level_db, 1),
            "output_noise_level_db": round(self.output_noise_level_db, 1),
            "noise_reduction_db": round(self.noise_reduction_db, 1),
            "improvement_percentage": round(self.improvement_percentage, 1),
            "voice_preservation_score": round(self.voice_preservation_score, 1),
            "input_snr_db": round(self.input_snr_db, 1),
            "output_snr_db": round(self.output_snr_db, 1),
            "overall_quality_score": round(self.overall_quality_score, 1),
        }


def compute_quality_score(
    original_path: str,
    processed_path: str,
) -> QualityReport:
    """
    Compare original and processed audio to compute quality metrics.

    Args:
        original_path: Path to the original (pre-processing) audio file
        processed_path: Path to the processed (post-pipeline) audio file

    Returns:
        QualityReport with all metrics
    """
    report = QualityReport()

    try:
        # Load both audio files
        original, sr_orig = sf.read(original_path, dtype="float32")
        processed, sr_proc = sf.read(processed_path, dtype="float32")

        # Ensure mono
        if original.ndim > 1:
            original = np.mean(original, axis=1)
        if processed.ndim > 1:
            processed = np.mean(processed, axis=1)

        # Use the common sample rate and length
        sr = sr_orig
        min_len = min(len(original), len(processed))
        original = original[:min_len]
        processed = processed[:min_len]

        # --- Measure noise levels from silent segments ---
        report.input_noise_level_db = _estimate_noise_level(original, sr)
        report.output_noise_level_db = _estimate_noise_level(processed, sr)

        # --- Noise reduction ---
        report.noise_reduction_db = report.input_noise_level_db - report.output_noise_level_db

        # Improvement percentage (how much noise was removed relative to input noise)
        if report.input_noise_level_db < 0:
            # Both are negative dB values; more negative = quieter noise floor
            # Improvement = how much further down the noise went
            noise_range = abs(report.input_noise_level_db)
            if noise_range > 0:
                improvement = (report.noise_reduction_db / noise_range) * 100
                report.improvement_percentage = float(np.clip(improvement, 0, 100))
            else:
                report.improvement_percentage = 0.0
        else:
            report.improvement_percentage = 0.0

        # --- SNR estimation ---
        report.input_snr_db = _estimate_snr(original, sr)
        report.output_snr_db = _estimate_snr(processed, sr)

        # --- Voice preservation score ---
        report.voice_preservation_score = _compute_voice_preservation(
            original, processed, sr
        )

        # --- Overall quality score ---
        # Weighted combination: noise reduction (40%) + voice preservation (40%) + SNR improvement (20%)
        snr_improvement_score = min(100, max(0, (report.output_snr_db - report.input_snr_db) * 5))
        noise_reduction_score = min(100, max(0, report.improvement_percentage))

        report.overall_quality_score = (
            noise_reduction_score * 0.4
            + report.voice_preservation_score * 0.4
            + snr_improvement_score * 0.2
        )

        logger.info(
            f"Quality report: noise={report.input_noise_level_db:.1f}→{report.output_noise_level_db:.1f}dB "
            f"(Δ{report.noise_reduction_db:.1f}dB, {report.improvement_percentage:.0f}%), "
            f"SNR={report.input_snr_db:.1f}→{report.output_snr_db:.1f}dB, "
            f"voice_preservation={report.voice_preservation_score:.0f}, "
            f"overall={report.overall_quality_score:.0f}"
        )

    except Exception as e:
        logger.error(f"Quality scoring failed: {e}", exc_info=True)

    return report


def _estimate_noise_level(audio: np.ndarray, sr: int) -> float:
    """
    Estimate noise floor level in dB from the quietest segments.
    Uses the 10th percentile of frame RMS as the noise floor estimate.
    """
    frame_length = int(0.02 * sr)  # 20ms frames
    hop_length = frame_length // 2

    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]

    # Use 10th percentile as noise floor (bottom 10% of energy)
    noise_rms = np.percentile(rms, 10)

    if noise_rms > 1e-10:
        return float(20 * np.log10(noise_rms))
    else:
        return -96.0  # Near digital silence


def _estimate_snr(audio: np.ndarray, sr: int) -> float:
    """
    Estimate signal-to-noise ratio.
    Signal = 90th percentile RMS, Noise = 10th percentile RMS.
    """
    frame_length = int(0.02 * sr)
    hop_length = frame_length // 2

    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]

    signal_rms = np.percentile(rms, 90)
    noise_rms = np.percentile(rms, 10)

    if noise_rms > 1e-10:
        return float(20 * np.log10(signal_rms / noise_rms))
    else:
        return 60.0


def _compute_voice_preservation(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int,
) -> float:
    """
    Compute voice preservation score (0–100).
    Measures how well the processing preserved the voice spectral characteristics
    in the speech frequency band (300–3400 Hz telephony band + extended to 6 kHz).
    """
    try:
        n_fft = 2048
        hop_length = 512

        # Compute spectral features for both
        orig_stft = np.abs(librosa.stft(original, n_fft=n_fft, hop_length=hop_length))
        proc_stft = np.abs(librosa.stft(processed, n_fft=n_fft, hop_length=hop_length))

        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Focus on voice band (300–6000 Hz)
        voice_mask = (freqs >= 300) & (freqs <= 6000)

        orig_voice = orig_stft[voice_mask, :]
        proc_voice = proc_stft[voice_mask, :]

        # Compare spectral envelopes (averaged over time)
        orig_envelope = np.mean(orig_voice, axis=1)
        proc_envelope = np.mean(proc_voice, axis=1)

        # Normalize envelopes
        orig_norm = orig_envelope / (np.max(orig_envelope) + 1e-10)
        proc_norm = proc_envelope / (np.max(proc_envelope) + 1e-10)

        # Compute spectral correlation (cosine similarity)
        dot_product = np.dot(orig_norm, proc_norm)
        norm_orig = np.linalg.norm(orig_norm)
        norm_proc = np.linalg.norm(proc_norm)

        if norm_orig > 1e-10 and norm_proc > 1e-10:
            correlation = dot_product / (norm_orig * norm_proc)
        else:
            correlation = 0.0

        # Also check spectral distortion (log spectral distance)
        orig_db = librosa.power_to_db(orig_envelope ** 2 + 1e-10)
        proc_db = librosa.power_to_db(proc_envelope ** 2 + 1e-10)
        lsd = np.sqrt(np.mean((orig_db - proc_db) ** 2))

        # Combine: high correlation = good, low LSD = good
        correlation_score = max(0, correlation) * 100
        lsd_score = max(0, 100 - lsd * 2)  # LSD of 0=perfect, 50+=terrible

        # Weighted average
        score = correlation_score * 0.6 + lsd_score * 0.4

        return float(np.clip(score, 0, 100))

    except Exception as e:
        logger.error(f"Voice preservation scoring failed: {e}")
        return 75.0  # Default moderate score on error

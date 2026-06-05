"""
Automatic noise detection and classification engine.
Analyzes spectral characteristics of audio to identify noise types
and estimate signal-to-noise ratio.

Supported noise types:
  - Fan noise (sustained low-frequency, low spectral variation)
  - AC noise (50/60 Hz tonal harmonics)
  - Traffic (broadband low-frequency rumble)
  - Wind (very low frequency turbulence)
  - Keyboard (short transient clicks in 1–8 kHz)
  - Hum (sharp peaks at 50/60 Hz + harmonics)
  - Hiss (elevated flat energy above 4 kHz)
  - Background conversations (speech-like spectral shape at lower energy)
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


class NoiseType(str, Enum):
    FAN = "fan"
    AC = "ac"
    TRAFFIC = "traffic"
    WIND = "wind"
    KEYBOARD = "keyboard"
    HUM = "hum"
    HISS = "hiss"
    BACKGROUND_CONVERSATION = "background_conversation"


@dataclass
class NoiseProfile:
    """Result of noise analysis on an audio file."""
    detected_types: dict[str, float] = field(default_factory=dict)  # type -> confidence (0-1)
    estimated_snr_db: float = 0.0
    noise_floor_db: float = -60.0
    dominant_noise: Optional[str] = None
    noise_spectral_profile: Optional[np.ndarray] = None  # for spectral subtraction

    def to_dict(self) -> dict:
        return {
            "detected_types": {k: round(v, 3) for k, v in self.detected_types.items()},
            "estimated_snr_db": round(self.estimated_snr_db, 1),
            "noise_floor_db": round(self.noise_floor_db, 1),
            "dominant_noise": self.dominant_noise,
        }


def classify_noise(audio: np.ndarray, sr: int) -> NoiseProfile:
    """
    Analyze audio and classify the types of background noise present.

    Args:
        audio: Audio signal as numpy array (mono, float32)
        sr: Sample rate

    Returns:
        NoiseProfile with detected noise types and confidence scores
    """
    profile = NoiseProfile()
    detected = {}

    try:
        # Compute STFT for spectral analysis
        n_fft = 2048
        hop_length = 512
        stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Compute power spectrum (average over time)
        power_spectrum = np.mean(stft ** 2, axis=1)
        power_db = librosa.power_to_db(power_spectrum + 1e-10)

        # --- Detect noise floor from quietest segments ---
        rms_frames = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)[0]
        quiet_threshold = np.percentile(rms_frames, 15)
        quiet_mask = rms_frames < quiet_threshold

        if np.any(quiet_mask):
            noise_spectrum = np.mean(stft[:, quiet_mask] ** 2, axis=1)
            noise_db = librosa.power_to_db(noise_spectrum + 1e-10)
            profile.noise_spectral_profile = noise_spectrum
            profile.noise_floor_db = float(np.median(noise_db))
        else:
            noise_spectrum = power_spectrum * 0.1
            noise_db = power_db - 10
            profile.noise_floor_db = float(np.median(noise_db))

        # --- Estimate SNR ---
        signal_rms = np.sqrt(np.mean(audio ** 2))
        if np.any(quiet_mask):
            quiet_indices = np.where(quiet_mask)[0]
            noise_samples = []
            for qi in quiet_indices:
                start = qi * hop_length
                end = min(start + n_fft, len(audio))
                if start < len(audio):
                    noise_samples.append(audio[start:end])
            if noise_samples:
                noise_concat = np.concatenate(noise_samples)
                noise_rms = np.sqrt(np.mean(noise_concat ** 2))
            else:
                noise_rms = signal_rms * 0.01
        else:
            noise_rms = signal_rms * 0.1

        if noise_rms > 1e-10:
            profile.estimated_snr_db = float(20 * np.log10(signal_rms / noise_rms))
        else:
            profile.estimated_snr_db = 60.0

        # ========================================
        # Noise type detection
        # ========================================

        # --- 1. Hum detection (50/60 Hz + harmonics) ---
        hum_confidence = _detect_hum(freqs, power_db, noise_db)
        if hum_confidence > 0.3:
            detected[NoiseType.HUM.value] = hum_confidence

        # --- 2. AC noise (similar to hum but broader tonal) ---
        ac_confidence = _detect_ac(freqs, power_db, stft, quiet_mask)
        if ac_confidence > 0.3:
            detected[NoiseType.AC.value] = ac_confidence

        # --- 3. Fan noise (sustained low-freq, low variation) ---
        fan_confidence = _detect_fan(freqs, stft, power_spectrum)
        if fan_confidence > 0.3:
            detected[NoiseType.FAN.value] = fan_confidence

        # --- 4. Wind noise (very low freq turbulence) ---
        wind_confidence = _detect_wind(freqs, stft, power_spectrum)
        if wind_confidence > 0.3:
            detected[NoiseType.WIND.value] = wind_confidence

        # --- 5. Traffic noise (broadband low rumble) ---
        traffic_confidence = _detect_traffic(freqs, stft, power_spectrum)
        if traffic_confidence > 0.3:
            detected[NoiseType.TRAFFIC.value] = traffic_confidence

        # --- 6. Keyboard clicks (transient 1-8 kHz) ---
        keyboard_confidence = _detect_keyboard(audio, sr, freqs, stft)
        if keyboard_confidence > 0.3:
            detected[NoiseType.KEYBOARD.value] = keyboard_confidence

        # --- 7. Hiss (flat elevated energy >4 kHz) ---
        hiss_confidence = _detect_hiss(freqs, noise_db)
        if hiss_confidence > 0.3:
            detected[NoiseType.HISS.value] = hiss_confidence

        # --- 8. Background conversations ---
        conversation_confidence = _detect_background_conversation(freqs, stft, quiet_mask, sr)
        if conversation_confidence > 0.3:
            detected[NoiseType.BACKGROUND_CONVERSATION.value] = conversation_confidence

        profile.detected_types = detected
        if detected:
            profile.dominant_noise = max(detected, key=detected.get)

        logger.info(
            f"Noise classification: SNR={profile.estimated_snr_db:.1f}dB, "
            f"floor={profile.noise_floor_db:.1f}dB, "
            f"types={profile.detected_types}, dominant={profile.dominant_noise}"
        )

    except Exception as e:
        logger.error(f"Noise classification failed: {e}", exc_info=True)

    return profile


def classify_noise_from_file(input_path: str) -> NoiseProfile:
    """Convenience: load audio from file and classify."""
    audio, sr = sf.read(input_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return classify_noise(audio, sr)


def detect_noise_floor(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract the noise-only spectral profile from the quietest segments.
    Used as a reference for spectral subtraction.
    """
    n_fft = 2048
    hop_length = 512
    stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    rms_frames = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)[0]

    quiet_threshold = np.percentile(rms_frames, 15)
    quiet_mask = rms_frames < quiet_threshold

    if np.any(quiet_mask):
        return np.mean(stft[:, quiet_mask] ** 2, axis=1)
    else:
        return np.mean(stft ** 2, axis=1) * 0.05


# ============================================================
# Individual noise type detectors
# ============================================================

def _detect_hum(freqs: np.ndarray, power_db: np.ndarray, noise_db: np.ndarray) -> float:
    """Detect electrical hum at 50/60 Hz and harmonics."""
    hum_freqs_50 = [50, 100, 150, 200, 250, 300]
    hum_freqs_60 = [60, 120, 180, 240, 300, 360]
    tolerance_hz = 5

    best_score = 0.0
    for hum_set in [hum_freqs_50, hum_freqs_60]:
        peaks_found = 0
        peak_prominence_sum = 0.0
        for hf in hum_set:
            mask = np.abs(freqs - hf) < tolerance_hz
            if np.any(mask):
                peak_power = np.max(noise_db[mask])
                # Check surrounding region for prominence
                surround_mask = (np.abs(freqs - hf) > tolerance_hz) & (np.abs(freqs - hf) < tolerance_hz * 4)
                if np.any(surround_mask):
                    surround_avg = np.mean(noise_db[surround_mask])
                    prominence = peak_power - surround_avg
                    if prominence > 6:  # at least 6dB above surroundings
                        peaks_found += 1
                        peak_prominence_sum += prominence
        if peaks_found >= 2:
            score = min(1.0, (peaks_found / 4.0) * (peak_prominence_sum / (peaks_found * 15)))
            best_score = max(best_score, score)

    return best_score


def _detect_ac(
    freqs: np.ndarray, power_db: np.ndarray,
    stft: np.ndarray, quiet_mask: np.ndarray
) -> float:
    """Detect AC/HVAC noise: steady broadband in 100–500 Hz."""
    band_mask = (freqs >= 100) & (freqs <= 500)
    if not np.any(band_mask):
        return 0.0

    band_stft = stft[band_mask, :]
    # AC noise is very steady over time
    temporal_std = np.std(band_stft, axis=1) / (np.mean(band_stft, axis=1) + 1e-10)
    steadiness = 1.0 - np.mean(temporal_std)
    steadiness = np.clip(steadiness, 0, 1)

    # Also check energy in this band during quiet segments
    if np.any(quiet_mask):
        quiet_energy = np.mean(band_stft[:, quiet_mask] ** 2)
        total_energy = np.mean(stft ** 2)
        energy_ratio = quiet_energy / (total_energy + 1e-10)
    else:
        energy_ratio = 0.0

    score = steadiness * 0.5 + min(1.0, energy_ratio * 10) * 0.5
    return float(np.clip(score, 0, 1))


def _detect_fan(freqs: np.ndarray, stft: np.ndarray, power_spectrum: np.ndarray) -> float:
    """Detect fan noise: sustained energy 100–500 Hz, low spectral variation."""
    band_mask = (freqs >= 80) & (freqs <= 600)
    if not np.any(band_mask):
        return 0.0

    band_power = power_spectrum[band_mask]
    total_power = np.sum(power_spectrum)
    band_ratio = np.sum(band_power) / (total_power + 1e-10)

    # Fan noise should be very steady
    band_stft = stft[band_mask, :]
    coeff_of_var = np.std(band_stft, axis=1) / (np.mean(band_stft, axis=1) + 1e-10)
    mean_cov = np.mean(coeff_of_var)
    steadiness = max(0, 1.0 - mean_cov)

    score = band_ratio * steadiness * 2.0
    return float(np.clip(score, 0, 1))


def _detect_wind(freqs: np.ndarray, stft: np.ndarray, power_spectrum: np.ndarray) -> float:
    """Detect wind noise: strong energy <100 Hz with high temporal variance."""
    band_mask = freqs < 120
    if not np.any(band_mask):
        return 0.0

    band_power = power_spectrum[band_mask]
    total_power = np.sum(power_spectrum)
    band_ratio = np.sum(band_power) / (total_power + 1e-10)

    # Wind has high temporal variance (gusts)
    band_stft = stft[band_mask, :]
    temporal_var = np.mean(np.std(band_stft, axis=1) / (np.mean(band_stft, axis=1) + 1e-10))

    score = band_ratio * temporal_var * 3.0
    return float(np.clip(score, 0, 1))


def _detect_traffic(freqs: np.ndarray, stft: np.ndarray, power_spectrum: np.ndarray) -> float:
    """Detect traffic noise: broadband rumble 20–300 Hz with intermittent bursts."""
    band_mask = (freqs >= 20) & (freqs <= 350)
    if not np.any(band_mask):
        return 0.0

    band_stft = stft[band_mask, :]
    # Look for intermittent energy bursts (not steady like fan)
    frame_energies = np.sum(band_stft ** 2, axis=0)
    burst_ratio = np.std(frame_energies) / (np.mean(frame_energies) + 1e-10)

    band_power = power_spectrum[band_mask]
    total_power = np.sum(power_spectrum)
    band_ratio = np.sum(band_power) / (total_power + 1e-10)

    # Traffic: moderate energy in low band + intermittent pattern
    score = band_ratio * burst_ratio * 2.0
    return float(np.clip(score, 0, 1))


def _detect_keyboard(
    audio: np.ndarray, sr: int,
    freqs: np.ndarray, stft: np.ndarray
) -> float:
    """Detect keyboard clicks: short transients in 1–8 kHz."""
    # Detect transients using onset strength
    try:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=512)
        onset_peaks = librosa.onset.onset_detect(
            y=audio, sr=sr, hop_length=512, backtrack=False, units='frames'
        )

        if len(onset_peaks) < 3:
            return 0.0

        # Keyboard clicks are very short and concentrated in high frequencies
        band_mask = (freqs >= 1000) & (freqs <= 8000)
        if not np.any(band_mask):
            return 0.0

        # Check if onset peaks are numerous and regularly spaced (typing pattern)
        peak_intervals = np.diff(onset_peaks)
        if len(peak_intervals) < 2:
            return 0.0

        # Typing: many onsets, relatively short duration each
        onset_rate = len(onset_peaks) / (len(audio) / sr)  # onsets per second
        if onset_rate > 2.0:  # at least 2 clicks per second
            score = min(1.0, onset_rate / 10.0)
            return float(score)
    except Exception:
        pass

    return 0.0


def _detect_hiss(freqs: np.ndarray, noise_db: np.ndarray) -> float:
    """Detect hiss: elevated flat energy above 4 kHz."""
    high_mask = freqs >= 4000
    mid_mask = (freqs >= 1000) & (freqs <= 3000)

    if not np.any(high_mask) or not np.any(mid_mask):
        return 0.0

    high_level = np.mean(noise_db[high_mask])
    mid_level = np.mean(noise_db[mid_mask])

    # Hiss: high-frequency energy close to or above mid-frequency
    diff = high_level - mid_level
    if diff > -6:  # within 6dB of mid-range = hiss present
        # Flatness of high-frequency region
        high_std = np.std(noise_db[high_mask])
        flatness = max(0, 1.0 - high_std / 10.0)
        score = (1.0 + diff / 12.0) * flatness
        return float(np.clip(score, 0, 1))

    return 0.0


def _detect_background_conversation(
    freqs: np.ndarray, stft: np.ndarray,
    quiet_mask: np.ndarray, sr: int
) -> float:
    """Detect background conversations: speech-like spectral energy at low levels."""
    # Speech energy is concentrated in 300–3400 Hz
    speech_mask = (freqs >= 300) & (freqs <= 3400)
    if not np.any(speech_mask) or not np.any(quiet_mask):
        return 0.0

    speech_band_stft = stft[speech_mask, :]
    # During "quiet" segments of main speaker, check for speech-like energy
    quiet_speech_energy = np.mean(speech_band_stft[:, quiet_mask] ** 2)
    total_quiet_energy = np.mean(stft[:, quiet_mask] ** 2) + 1e-10

    # Background speech has energy concentrated in speech band even during main silence
    speech_ratio = quiet_speech_energy / total_quiet_energy

    # Also check for spectral modulation (speech has ~4 Hz modulation)
    if speech_ratio > 0.3:
        energy_over_time = np.sum(speech_band_stft[:, quiet_mask] ** 2, axis=0)
        if len(energy_over_time) > 10:
            # Check temporal modulation
            modulation = np.std(energy_over_time) / (np.mean(energy_over_time) + 1e-10)
            score = speech_ratio * min(1.0, modulation * 2.0)
            return float(np.clip(score, 0, 1))

    return 0.0

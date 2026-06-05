"""
Advanced audio filters for targeting specific noise profiles.
Uses numpy, scipy, and librosa to implement actual, high-quality audio DSP.

Enhanced with adaptive noise suppression strategies that apply
targeted filtering based on classified noise types.
"""
import logging
import numpy as np
import scipy.signal as signal
import soundfile as sf
import librosa

logger = logging.getLogger(__name__)


def apply_breath_remover(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Remove breath sounds from audio.
    Breaths are low-energy, high-frequency/noise-like segments between speech.
    Uses energy envelope and spectral roll-off to identify and attenuate breath segments.
    """
    logger.info("Applying breath remover filter...")
    try:
        # Frame size and hop length for short-time analysis
        frame_len = int(0.02 * sr)  # 20ms
        hop_len = int(0.01 * sr)   # 10ms

        # Compute RMS energy envelope
        rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]

        # Compute Zero Crossing Rate (ZCR) - breath/friction has high ZCR
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_len, hop_length=hop_len)[0]

        # Normalize features
        rms_norm = rms / (np.max(rms) + 1e-6)

        # Thresholds
        # Breaths are typically quiet (low RMS) but have high ZCR compared to silence
        energy_thresh = 0.05
        zcr_thresh = 0.12

        # Create attenuation gain mask per frame
        gains = np.ones_like(rms)
        for i in range(len(rms)):
            if rms_norm[i] < energy_thresh and zcr[i] > zcr_thresh:
                # Identified as breath/hiss pause -> attenuate by 18dB (factor of ~0.12)
                gains[i] = 0.12
            elif rms_norm[i] < 0.01:
                # Near silence -> gate completely
                gains[i] = 0.05

        # Smooth the gains mask to prevent clicking/artifacts
        gains_smooth = signal.medfilt(gains, kernel_size=15)

        # Interpolate frame-wise gains back to sample-wise level
        gains_samples = np.interp(
            np.arange(len(y)),
            np.arange(len(gains_smooth)) * hop_len,
            gains_smooth,
            right=1.0
        )

        return y * gains_samples
    except Exception as e:
        logger.error(f"Error in breath remover: {e}")
        return y


def apply_mouth_sounds_remover(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Remove mouth sounds (clicks, saliva pops, transient ticks).
    Detects sudden, short spikes in the signal's derivative and replaces them using median filtering.
    """
    logger.info("Applying mouth sounds / click remover filter...")
    try:
        # Find local transients (spikes in absolute derivative)
        diff = np.abs(np.diff(y))
        threshold = np.median(diff) * 6.0  # Threshold for clicks

        y_clean = np.copy(y)
        click_indices = np.where(diff > threshold)[0]

        # Replace detected clicks with median filtered version
        window_size = int(0.002 * sr)  # 2ms window for click replacement
        if window_size % 2 == 0:
            window_size += 1

        for idx in click_indices:
            start = max(0, idx - window_size // 2)
            end = min(len(y), idx + window_size // 2)
            if start < end:
                y_clean[idx] = np.median(y[start:end])

        return y_clean
    except Exception as e:
        logger.error(f"Error in mouth sounds remover: {e}")
        return y


def apply_background_music_attenuator(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Attenuate background music.
    Uses Harmonic-Percussive Source Separation (HPSS).
    Music is mostly harmonic (steady tones), whereas voice is a mixture but heavily percussive
    in terms of speech consonants and transient dynamics. Suppressing the harmonic component
    reduces melodic background tracks.
    """
    logger.info("Applying background music attenuator...")
    try:
        # Perform HPSS
        stft = librosa.stft(y)
        harmonic, percussive = librosa.decompose.hpss(stft, margin=1.5)

        # Reconstruct the percussive/vocal dominant part
        # Keep percussive component and mix in only a small fraction of harmonic to preserve vocal clarity
        y_percussive = librosa.istft(percussive)
        y_harmonic = librosa.istft(harmonic)

        # Mix them back with harmonic attenuated by 12dB (factor of 0.25)
        return y_percussive + (0.25 * y_harmonic)
    except Exception as e:
        logger.error(f"Error in background music attenuator: {e}")
        return y


def apply_low_cut_filter(y: np.ndarray, sr: int, cutoff: float = 120.0) -> np.ndarray:
    """Apply a butterworth high-pass filter to remove low-frequency rumble (helicopter, wind)."""
    try:
        nyq = 0.5 * sr
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
        return signal.filtfilt(b, a, y)
    except Exception as e:
        logger.error(f"Error in highpass filter: {e}")
        return y


def apply_high_cut_filter(y: np.ndarray, sr: int, cutoff: float = 6500.0) -> np.ndarray:
    """Apply a butterworth low-pass filter to remove high-frequency sizzle/hiss (water, steam)."""
    try:
        nyq = 0.5 * sr
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
        return signal.filtfilt(b, a, y)
    except Exception as e:
        logger.error(f"Error in lowpass filter: {e}")
        return y


def apply_adaptive_suppression(
    audio: np.ndarray,
    sr: int,
    noise_types: dict[str, float],
    quality_mode: str = "balanced",
) -> np.ndarray:
    """
    Apply targeted filtering based on classified noise types.
    Suppression strength is scaled by quality mode's adaptive_scale.

    Voice preservation: all confidence values are multiplied by the
    quality mode's adaptive_scale (0.3 for light, 0.5 for balanced)
    to prevent over-suppression that causes metallic artifacts.

    Args:
        audio: Audio signal (mono, float32)
        sr: Sample rate
        noise_types: Dict of noise_type -> confidence (from NoiseClassifier)
        quality_mode: Quality mode string

    Returns:
        Processed audio with type-specific noise suppression
    """
    from app.config import get_mode_params

    if not noise_types:
        logger.info("No noise types detected, skipping adaptive suppression")
        return audio

    params = get_mode_params(quality_mode)
    adaptive_scale = params.get("adaptive_scale", 0.5)

    logger.info(
        f"Adaptive suppression: types={list(noise_types.keys())}, "
        f"mode={quality_mode}, scale={adaptive_scale}"
    )
    y = audio.copy()

    for noise_type, raw_confidence in sorted(noise_types.items(), key=lambda x: -x[1]):
        # Scale confidence by quality mode
        confidence = raw_confidence * adaptive_scale
        if confidence < 0.15:  # Skip very weak detections
            continue

        try:
            if noise_type == "hum":
                y = _suppress_hum(y, sr, confidence)
            elif noise_type == "ac":
                y = _suppress_ac(y, sr, confidence)
            elif noise_type == "fan":
                y = _suppress_fan(y, sr, confidence)
            elif noise_type == "wind":
                y = _suppress_wind(y, sr, confidence)
            elif noise_type == "traffic":
                y = _suppress_traffic(y, sr, confidence)
                # Also apply horn suppression, as horns are part of traffic noise
                y = _suppress_horn(y, sr, confidence)
            elif noise_type == "keyboard":
                y = _suppress_keyboard(y, sr, confidence)
            elif noise_type == "hiss":
                y = _suppress_hiss(y, sr, confidence)
            elif noise_type == "background_conversation":
                y = _suppress_background_conversation(y, sr, confidence)
        except Exception as e:
            logger.warning(f"Failed to suppress {noise_type}: {e}")

    return np.clip(y, -1.0, 1.0)


# ============================================================
# Type-specific suppression strategies
# ============================================================

def _suppress_hum(y: np.ndarray, sr: int, confidence: float) -> np.ndarray:
    """Suppress electrical hum using adaptive notch filters at 50/60 Hz harmonics."""
    logger.info(f"Suppressing hum (confidence={confidence:.2f})")
    nyq = sr / 2.0

    # Apply notch filters at both 50 Hz and 60 Hz harmonics
    gain_db = -12 * confidence  # Scale suppression by confidence
    for base_freq in [50, 60]:
        for harmonic in range(1, 7):  # Up to 6th harmonic
            freq = base_freq * harmonic
            if freq >= nyq:
                break
            # Narrow notch filter
            w0 = freq / nyq
            Q = 30  # Very narrow Q for precise notch
            b, a = signal.iirnotch(w0, Q)
            y = signal.filtfilt(b, a, y)

    return y


def _suppress_ac(y: np.ndarray, sr: int, confidence: float) -> np.ndarray:
    """Suppress AC/HVAC noise: broadband attenuation in 100–500 Hz."""
    logger.info(f"Suppressing AC noise (confidence={confidence:.2f})")

    # Use spectral subtraction in the AC noise band
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    ac_band = (freqs >= 100) & (freqs <= 500)

    # Estimate noise floor in AC band from quietest frames
    frame_energies = np.sum(magnitude ** 2, axis=0)
    quiet_mask = frame_energies < np.percentile(frame_energies, 20)

    if np.any(quiet_mask):
        noise_estimate = np.mean(magnitude[:, quiet_mask], axis=1, keepdims=True)
    else:
        noise_estimate = np.min(magnitude, axis=1, keepdims=True) * 0.5

    # Spectral subtraction only in AC band
    suppression = np.ones_like(magnitude)
    suppression[ac_band, :] = np.maximum(
        0.05,  # noise floor
        1.0 - confidence * noise_estimate[ac_band, :] / (magnitude[ac_band, :] + 1e-10)
    )

    cleaned_stft = magnitude * suppression * np.exp(1j * phase)
    return librosa.istft(cleaned_stft, hop_length=hop_length, length=len(y))


def _suppress_fan(y: np.ndarray, sr: int, confidence: float) -> np.ndarray:
    """Suppress fan noise: spectral subtraction in 80–600 Hz band."""
    logger.info(f"Suppressing fan noise (confidence={confidence:.2f})")

    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    fan_band = (freqs >= 80) & (freqs <= 600)

    # Estimate steady-state noise (fan is very consistent)
    noise_estimate = np.median(magnitude, axis=1, keepdims=True)

    suppression = np.ones_like(magnitude)
    subtraction_factor = 1.5 * confidence
    suppression[fan_band, :] = np.maximum(
        0.08,
        1.0 - subtraction_factor * noise_estimate[fan_band, :] / (magnitude[fan_band, :] + 1e-10)
    )

    cleaned_stft = magnitude * suppression * np.exp(1j * phase)
    return librosa.istft(cleaned_stft, hop_length=hop_length, length=len(y))


def _suppress_wind(y: np.ndarray, sr: int, confidence: float) -> np.ndarray:
    """Suppress wind noise: aggressive highpass + spectral subtraction below 120 Hz."""
    logger.info(f"Suppressing wind noise (confidence={confidence:.2f})")

    # Steep highpass filter scaled by confidence
    cutoff = 80 + 60 * confidence  # 80–140 Hz depending on severity
    y = apply_low_cut_filter(y, sr, cutoff=cutoff)

    # Additional spectral subtraction in very low frequencies
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    wind_band = freqs < 150

    noise_estimate = np.percentile(magnitude, 50, axis=1, keepdims=True)

    suppression = np.ones_like(magnitude)
    suppression[wind_band, :] = np.maximum(
        0.02,
        1.0 - 2.0 * confidence * noise_estimate[wind_band, :] / (magnitude[wind_band, :] + 1e-10)
    )

    cleaned_stft = magnitude * suppression * np.exp(1j * phase)
    return librosa.istft(cleaned_stft, hop_length=hop_length, length=len(y))


def _suppress_traffic(y: np.ndarray, sr: int, confidence: float) -> np.ndarray:
    """Suppress traffic noise: broadband low-frequency reduction with burst detection."""
    logger.info(f"Suppressing traffic noise (confidence={confidence:.2f})")

    # Moderate highpass
    y = apply_low_cut_filter(y, sr, cutoff=100 + 50 * confidence)

    # Spectral subtraction in traffic band
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    traffic_band = (freqs >= 20) & (freqs <= 350)

    noise_estimate = np.percentile(magnitude, 30, axis=1, keepdims=True)

    suppression = np.ones_like(magnitude)
    suppression[traffic_band, :] = np.maximum(
        0.1,
        1.0 - confidence * noise_estimate[traffic_band, :] / (magnitude[traffic_band, :] + 1e-10)
    )

    cleaned_stft = magnitude * suppression * np.exp(1j * phase)
    return librosa.istft(cleaned_stft, hop_length=hop_length, length=len(y))


def _suppress_horn(y: np.ndarray, sr: int, confidence: float) -> np.ndarray:
    """
    Suppress car horns: tonal, high-amplitude mid-frequency sounds (350 Hz - 2000 Hz).
    Uses a spectral subtraction approach with high subtraction factor in the horn band.
    """
    logger.info(f"Suppressing car horn noise (confidence={confidence:.2f})")
    
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    # Horn band: 350 Hz to 2000 Hz
    horn_band = (freqs >= 350) & (freqs <= 2000)
    
    # Estimate the local noise floor in the horn band
    # Horns are transient but have very high energy peaks
    # We use median over time to find the background level, and target high-energy transients in this band
    median_spec = np.median(magnitude, axis=1, keepdims=True)
    
    # Find frames where the energy in the horn band is significantly higher than the median
    # This indicates a honk/beep transient
    suppression = np.ones_like(magnitude)
    
    for f_idx in range(magnitude.shape[1]):
        frame = magnitude[:, f_idx]
        # Check if the average energy in the horn band is higher than 2x the median background
        if np.mean(frame[horn_band]) > 2.0 * np.mean(median_spec[horn_band]):
            # This frame likely contains a horn honk
            # Apply strong subtraction to peaks in the horn band
            for bin_idx in np.where(horn_band)[0]:
                if frame[bin_idx] > 2.5 * median_spec[bin_idx]:
                    # Attenuate the peak by subtracting noise estimate
                    attenuation = confidence * 0.90 # up to 90% attenuation
                    suppression[bin_idx, f_idx] = max(0.05, 1.0 - attenuation)
                    
    cleaned_stft = magnitude * suppression * np.exp(1j * phase)
    return librosa.istft(cleaned_stft, hop_length=hop_length, length=len(y))


def _suppress_keyboard(y: np.ndarray, sr: int, confidence: float) -> np.ndarray:
    """Suppress keyboard clicks using transient detection and interpolation."""
    logger.info(f"Suppressing keyboard clicks (confidence={confidence:.2f})")

    y_clean = np.copy(y)

    # Detect transients (sharp energy spikes)
    diff = np.abs(np.diff(y))
    threshold = np.percentile(diff, 98) * (2.0 - confidence)

    click_indices = np.where(diff > threshold)[0]

    # Interpolate over detected clicks
    click_window = int(0.003 * sr)  # 3ms window
    if click_window % 2 == 0:
        click_window += 1

    for idx in click_indices:
        start = max(0, idx - click_window)
        end = min(len(y) - 1, idx + click_window)
        if start < end and end - start > 2:
            # Linear interpolation over click
            y_clean[start:end] = np.linspace(y_clean[start], y_clean[end], end - start)

    return y_clean


def _suppress_hiss(y: np.ndarray, sr: int, confidence: float) -> np.ndarray:
    """Suppress hiss: adaptive Wiener-like filter above 4 kHz."""
    logger.info(f"Suppressing hiss (confidence={confidence:.2f})")

    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    hiss_band = freqs >= 4000

    # Estimate noise floor in hiss band
    noise_power = np.percentile(magnitude[hiss_band, :] ** 2, 25, axis=1, keepdims=True)

    # Wiener-like filter: gain = signal_power / (signal_power + noise_power)
    signal_power = magnitude[hiss_band, :] ** 2
    wiener_gain = signal_power / (signal_power + noise_power * confidence * 3.0 + 1e-10)
    wiener_gain = np.clip(wiener_gain, 0.05, 1.0)

    suppression = np.ones_like(magnitude)
    suppression[hiss_band, :] = np.sqrt(wiener_gain)

    cleaned_stft = magnitude * suppression * np.exp(1j * phase)
    return librosa.istft(cleaned_stft, hop_length=hop_length, length=len(y))


def _suppress_background_conversation(y: np.ndarray, sr: int, confidence: float) -> np.ndarray:
    """
    Suppress background conversations using spectral masking.
    Uses the primary voice spectral envelope to mask competing speech.
    """
    logger.info(f"Suppressing background conversation (confidence={confidence:.2f})")

    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    speech_band = (freqs >= 300) & (freqs <= 3400)

    # Identify loud (primary speaker) vs quiet (background) frames
    frame_energies = np.sum(magnitude[speech_band, :] ** 2, axis=0)
    energy_threshold = np.percentile(frame_energies, 60)

    loud_frames = frame_energies > energy_threshold
    quiet_frames = ~loud_frames

    # Build spectral mask: during quiet frames, suppress speech band
    suppression = np.ones_like(magnitude)
    if np.any(quiet_frames):
        # Estimate background speech level from quiet frames
        bg_level = np.mean(magnitude[speech_band][:, quiet_frames], axis=1, keepdims=True)
        primary_level = np.mean(magnitude[speech_band][:, loud_frames], axis=1, keepdims=True) if np.any(loud_frames) else bg_level

        # Suppress speech band in quiet frames proportionally
        for i in range(magnitude.shape[1]):
            if quiet_frames[i]:
                attenuation = confidence * 0.7  # Don't over-suppress
                suppression[speech_band, i] = max(0.15, 1.0 - attenuation)

    cleaned_stft = magnitude * suppression * np.exp(1j * phase)
    return librosa.istft(cleaned_stft, hop_length=hop_length, length=len(y))


def process_custom_filters(
    input_path: str,
    output_path: str,
    breath_remover: bool = False,
    mouth_sounds_remover: bool = False,
    background_music: bool = False,
    helicopter: bool = False,
    water: bool = False,
    dog_barking: bool = False,
    restaurant_chatter: bool = False,
) -> str:
    """
    Run custom DSP filters on the audio depending on user selections.
    Usually applied after DeepFilterNet / spectral gating.
    """
    # If no options selected, just copy
    any_selected = (
        breath_remover or mouth_sounds_remover or background_music or
        helicopter or water or dog_barking or restaurant_chatter
    )
    if not any_selected:
        import shutil
        shutil.copy(input_path, output_path)
        return output_path

    logger.info(
        f"Processing custom filters: breath={breath_remover}, mouth_sounds={mouth_sounds_remover}, "
        f"music={background_music}, helicopter={helicopter}, water={water}, dog={dog_barking}, chatter={restaurant_chatter}"
    )

    try:
        y, sr = sf.read(input_path, dtype="float32")

        # 1. Mouth clicks (transients)
        if mouth_sounds_remover:
            y = apply_mouth_sounds_remover(y, sr)

        # 2. Helicopter / Low-end rumble
        if helicopter:
            # Steep highpass to cut low frequency rotar rumble below 150Hz
            y = apply_low_cut_filter(y, sr, cutoff=150.0)

        # 3. Water / Rain hiss
        if water:
            # Lowpass at 6kHz to remove water sizzle
            y = apply_high_cut_filter(y, sr, cutoff=6000.0)

        # 4. Background music
        if background_music:
            y = apply_background_music_attenuator(y, sr)

        # 5. Breath sounds
        if breath_remover:
            y = apply_breath_remover(y, sr)

        # 6. Dog Barking / Chatter adjustments (if needed, but DeepFilterNet handles most of it.
        # We can apply a noise gate or moderate bandpass to clean up further)
        if dog_barking or restaurant_chatter:
            # Speech bandpass filter: voice resides primarily in 100Hz - 7.5kHz
            y = apply_low_cut_filter(y, sr, cutoff=100.0)
            y = apply_high_cut_filter(y, sr, cutoff=7500.0)

        # Clip to prevent artifacts
        y = np.clip(y, -1.0, 1.0)

        sf.write(output_path, y, sr, subtype="PCM_16")
        return output_path
    except Exception as e:
        logger.error(f"Custom filter processing failed: {e}")
        import shutil
        shutil.copy(input_path, output_path)
        return output_path

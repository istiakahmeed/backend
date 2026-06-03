"""
Advanced audio filters for targeting specific noise profiles.
Uses numpy, scipy, and librosa to implement actual, high-quality audio DSP.
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

"""
Voice Activity Detection (VAD) service.
Separates speech from non-speech regions for targeted noise reduction.

Tuned for GENEROUS speech detection:
  - Lower energy threshold to catch quiet syllables
  - Longer minimum silence duration to prevent choppy cuts
  - Wider crossfade (30ms) for smooth transitions
  - Errs on the side of classifying as speech (safer for voice preservation)
"""
import logging
from typing import Optional

import numpy as np
import librosa
import soundfile as sf

from app.config import (
    VAD_AGGRESSIVENESS,
    VAD_MIN_SPEECH_MS,
    VAD_MIN_SILENCE_MS,
    VAD_ENERGY_THRESHOLD,
    VAD_FRAME_DURATION_MS,
    VAD_CROSSFADE_MS,
)

logger = logging.getLogger(__name__)


def detect_speech_segments(
    audio: np.ndarray,
    sr: int,
    aggressiveness: Optional[int] = None,
) -> list[tuple[int, int]]:
    """
    Detect speech segments using WebRTC VAD if available,
    falling back to librosa-based custom VAD.
    """
    if aggressiveness is None:
        aggressiveness = VAD_AGGRESSIVENESS

    # We only use webrtcvad if the sample rate is supported
    use_webrtc = False
    if sr in [8000, 16000, 32000, 48000]:
        try:
            import webrtcvad
            use_webrtc = True
        except ImportError:
            logger.warning("webrtcvad not installed, falling back to librosa VAD")

    if use_webrtc:
        import webrtcvad
        # WebRTC VAD aggressiveness: 0 (most permissive) to 3 (most aggressive)
        webrtc_aggr = max(0, min(3, aggressiveness))
        vad = webrtcvad.Vad(webrtc_aggr)

        # webrtcvad expects 16-bit PCM bytes
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        
        frame_duration_ms = 20
        frame_samples = int(sr * frame_duration_ms / 1000)
        
        # Pad to be a multiple of frame_samples
        remainder = len(audio_int16) % frame_samples
        if remainder > 0:
            audio_int16 = np.pad(audio_int16, (0, frame_samples - remainder))
            
        n_frames = len(audio_int16) // frame_samples
        speech_frames = np.zeros(n_frames, dtype=bool)
        
        for i in range(n_frames):
            frame = audio_int16[i * frame_samples : (i + 1) * frame_samples]
            try:
                speech_frames[i] = vad.is_speech(frame.tobytes(), sr)
            except Exception as e:
                logger.error(f"webrtcvad frame processing failed: {e}")
                speech_frames[i] = True # Default to speech on error to avoid cutting voice

        # ============================================================
        # Professional Gating: Look-Ahead & Hold
        # ============================================================
        # 1. Look-ahead (40ms = 2 frames): open gate slightly before speech starts
        look_ahead_frames = 2
        smoothed = speech_frames.copy()
        for i in range(len(speech_frames)):
            if any(speech_frames[i : min(len(speech_frames), i + look_ahead_frames + 1)]):
                smoothed[i] = True
        speech_frames = smoothed

        # 2. Hold time (160ms = 8 frames): keep gate open after speech ends
        hold_frames = 8
        hold_counter = 0
        for i in range(len(speech_frames)):
            if speech_frames[i]:
                hold_counter = hold_frames
            else:
                if hold_counter > 0:
                    speech_frames[i] = True
                    hold_counter -= 1

        # 3. Standard smoothing
        min_speech_frames = int(VAD_MIN_SPEECH_MS / 20)
        min_silence_frames = int(VAD_MIN_SILENCE_MS / 20)
        speech_frames = _smooth_vad(speech_frames, min_speech_frames, min_silence_frames)

        # Convert frames to sample segments
        segments = []
        in_speech = False
        start = 0
        for i in range(n_frames):
            if speech_frames[i] and not in_speech:
                start = i * frame_samples
                in_speech = True
            elif not speech_frames[i] and in_speech:
                end = min(i * frame_samples, len(audio))
                segments.append((start, end))
                in_speech = False
        if in_speech:
            segments.append((start, len(audio)))
    else:
        # Fallback to librosa custom VAD
        frame_samples = int(VAD_FRAME_DURATION_MS * sr / 1000)
        hop_samples = frame_samples // 2

        rms = librosa.feature.rms(
            y=audio, frame_length=frame_samples, hop_length=hop_samples
        )[0]
        zcr = librosa.feature.zero_crossing_rate(
            y=audio, frame_length=frame_samples, hop_length=hop_samples
        )[0]
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr, n_fft=frame_samples, hop_length=hop_samples
        )[0]

        rms_norm = rms / (np.max(rms) + 1e-10)
        base_threshold = VAD_ENERGY_THRESHOLD
        threshold_multipliers = {0: 0.4, 1: 0.6, 2: 0.8, 3: 1.2}
        threshold = base_threshold * threshold_multipliers.get(aggressiveness, 0.6)

        energy_p15 = np.percentile(rms_norm, 15)
        energy_p80 = np.percentile(rms_norm, 80)
        if energy_p80 > 0:
            adaptive_threshold = energy_p15 + (energy_p80 - energy_p15) * 0.10
            threshold = max(threshold, adaptive_threshold)

        n_frames = len(rms)
        speech_frames = np.zeros(n_frames, dtype=bool)

        zcr_max = 0.35
        centroid_low = 200
        centroid_high = 6000

        for i in range(n_frames):
            is_speech = (
                rms_norm[i] > threshold
                and zcr[i] < zcr_max
                and centroid_low < centroid[i] < centroid_high
            )
            speech_frames[i] = is_speech

        high_energy_mask = rms_norm > (threshold * 2.5)
        speech_frames = speech_frames | high_energy_mask

        min_speech_frames = int(VAD_MIN_SPEECH_MS * sr / (hop_samples * 1000))
        min_silence_frames = int(VAD_MIN_SILENCE_MS * sr / (hop_samples * 1000))
        speech_frames = _smooth_vad(speech_frames, min_speech_frames, min_silence_frames)

        segments = []
        in_speech = False
        start = 0
        for i in range(n_frames):
            if speech_frames[i] and not in_speech:
                start = i * hop_samples
                in_speech = True
            elif not speech_frames[i] and in_speech:
                end = min(i * hop_samples, len(audio))
                segments.append((start, end))
                in_speech = False
        if in_speech:
            segments.append((start, len(audio)))

    total_speech = sum(e - s for s, e in segments)
    speech_pct = (total_speech / len(audio)) * 100 if len(audio) > 0 else 0

    logger.info(
        f"VAD ({'WebRTC' if use_webrtc else 'Librosa'}): {len(segments)} segments, "
        f"{speech_pct:.1f}% speech"
    )

    return segments


def create_speech_mask(
    audio: np.ndarray,
    sr: int,
    aggressiveness: Optional[int] = None,
    crossfade_ms: Optional[int] = None,
) -> np.ndarray:
    """
    Create per-sample mask with wide crossfades for smooth transitions.

    Returns float array: 1.0 = speech, 0.0 = non-speech.
    Crossfade prevents choppy artifacts at boundaries.
    """
    if crossfade_ms is None:
        crossfade_ms = VAD_CROSSFADE_MS

    segments = detect_speech_segments(audio, sr, aggressiveness)
    mask = np.zeros(len(audio), dtype=np.float32)

    for start, end in segments:
        mask[start:end] = 1.0

    # Apply wide crossfade smoothing at boundaries
    crossfade_samples = int(crossfade_ms * sr / 1000)
    if crossfade_samples > 1:
        ramp_up = np.linspace(0, 1, crossfade_samples, dtype=np.float32)
        ramp_down = np.linspace(1, 0, crossfade_samples, dtype=np.float32)

        for start, end in segments:
            # Ramp up at speech start
            ramp_start = max(0, start - crossfade_samples)
            ramp_end = min(len(audio), ramp_start + crossfade_samples)
            actual_len = ramp_end - ramp_start
            if actual_len > 0:
                mask[ramp_start:ramp_end] = np.maximum(
                    mask[ramp_start:ramp_end],
                    ramp_up[:actual_len]
                )

            # Ramp down at speech end
            ramp_start = end
            ramp_end = min(len(audio), end + crossfade_samples)
            actual_len = ramp_end - ramp_start
            if actual_len > 0:
                mask[ramp_start:ramp_end] = np.maximum(
                    mask[ramp_start:ramp_end],
                    ramp_down[:actual_len]
                )

    return mask


def create_speech_mask_from_file(
    input_path: str,
    aggressiveness: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load audio from file and create speech mask."""
    audio, sr = sf.read(input_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    mask = create_speech_mask(audio, sr, aggressiveness)
    return mask, audio, sr


def _smooth_vad(
    frames: np.ndarray,
    min_speech: int,
    min_silence: int,
) -> np.ndarray:
    """
    Smooth VAD decisions:
    - Fill short silence gaps (prevents choppy speech)
    - Remove too-short speech bursts (false positives)
    """
    result = frames.copy()

    # Fill short silence gaps FIRST (merge nearby speech)
    in_silence = False
    gap_start = 0
    for i in range(len(result)):
        if not result[i]:
            if not in_silence:
                gap_start = i
                in_silence = True
        else:
            if in_silence:
                gap_len = i - gap_start
                if gap_len < min_silence:
                    result[gap_start:i] = True
                in_silence = False

    # Remove too-short speech segments (likely noise bursts)
    in_speech = False
    seg_start = 0
    for i in range(len(result)):
        if result[i]:
            if not in_speech:
                seg_start = i
                in_speech = True
        else:
            if in_speech:
                seg_len = i - seg_start
                if seg_len < min_speech:
                    result[seg_start:i] = False
                in_speech = False

    return result

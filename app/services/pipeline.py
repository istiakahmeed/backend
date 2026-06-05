"""
Audio processing pipeline orchestrator.
Manages the full 8-stage processing flow from upload to podcast-quality output.

Voice-preservation-first philosophy:
  - Default "balanced" mode prioritizes natural speech over aggressive noise removal
  - Quality modes: light / balanced (default) / strong / maximum
  - A small amount of residual noise is acceptable; robotic/metallic voice is NOT
  - Low-confidence noise regions are left untouched to preserve voice
"""
import os
import uuid
import time
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from app.config import (
    TEMP_DIR, FILE_TTL_SECONDS, QUALITY_SCORE_ENABLED,
    QualityMode, DEFAULT_QUALITY_MODE, get_mode_params,
)

logger = logging.getLogger(__name__)


class ProcessingStep(str, Enum):
    QUEUED = "queued"
    PREPROCESSING = "preprocessing"
    NOISE_PROFILING = "noise_profiling"
    DEEPFILTER = "deepfilter"
    RNNOISE = "rnnoise"
    SPECTRAL_GATING = "spectral_gating"
    ADAPTIVE_SUPPRESSION = "adaptive_suppression"
    NORMALIZING = "normalizing"
    FINAL_ENHANCEMENT = "final_enhancement"
    EXPORTING = "exporting"
    COMPLETE = "complete"
    ERROR = "error"


STEP_DESCRIPTIONS = {
    ProcessingStep.QUEUED: "Waiting in queue...",
    ProcessingStep.PREPROCESSING: "Preprocessing audio...",
    ProcessingStep.NOISE_PROFILING: "Analyzing noise profile...",
    ProcessingStep.DEEPFILTER: "AI speech enhancement (DeepFilterNet 3)...",
    ProcessingStep.RNNOISE: "RNNoise denoising...",
    ProcessingStep.SPECTRAL_GATING: "Spectral gating (voice-aware)...",
    ProcessingStep.ADAPTIVE_SUPPRESSION: "Targeted noise suppression...",
    ProcessingStep.NORMALIZING: "Normalizing loudness...",
    ProcessingStep.FINAL_ENHANCEMENT: "Final voice polish...",
    ProcessingStep.EXPORTING: "Preparing output...",
    ProcessingStep.COMPLETE: "Processing complete!",
    ProcessingStep.ERROR: "An error occurred",
}

STEP_PROGRESS = {
    ProcessingStep.QUEUED: 0,
    ProcessingStep.PREPROCESSING: 5,
    ProcessingStep.NOISE_PROFILING: 12,
    ProcessingStep.DEEPFILTER: 25,
    ProcessingStep.RNNOISE: 38,
    ProcessingStep.SPECTRAL_GATING: 50,
    ProcessingStep.ADAPTIVE_SUPPRESSION: 62,
    ProcessingStep.NORMALIZING: 75,
    ProcessingStep.FINAL_ENHANCEMENT: 85,
    ProcessingStep.EXPORTING: 95,
    ProcessingStep.COMPLETE: 100,
    ProcessingStep.ERROR: 0,
}


@dataclass
class Job:
    """Represents a single audio processing job."""
    job_id: str
    original_filename: str
    input_path: str
    step: ProcessingStep = ProcessingStep.QUEUED
    progress: int = 0
    description: str = ""
    error_message: Optional[str] = None
    output_path: Optional[str] = None
    original_url_path: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    file_size_bytes: int = 0
    duration_seconds: float = 0.0

    # Custom filters options
    breath_remover: bool = False
    mouth_sounds_remover: bool = False
    background_music: bool = False
    helicopter: bool = False
    water: bool = False
    dog_barking: bool = False
    restaurant_chatter: bool = False

    # Quality mode (replaces max_noise_removal boolean)
    quality_mode: str = DEFAULT_QUALITY_MODE
    max_noise_removal: bool = False  # Legacy compat — maps to "maximum" mode

    # Results from pipeline
    quality_report: Optional[dict] = None
    noise_profile: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "original_filename": self.original_filename,
            "step": self.step.value,
            "progress": self.progress,
            "description": self.description,
            "error_message": self.error_message,
            "has_output": self.output_path is not None,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "file_size_bytes": self.file_size_bytes,
            "duration_seconds": self.duration_seconds,
            "quality_report": self.quality_report,
            "noise_profile": self.noise_profile,
            "options": {
                "breath_remover": self.breath_remover,
                "mouth_sounds_remover": self.mouth_sounds_remover,
                "background_music": self.background_music,
                "helicopter": self.helicopter,
                "water": self.water,
                "dog_barking": self.dog_barking,
                "restaurant_chatter": self.restaurant_chatter,
                "quality_mode": self.quality_mode,
                "max_noise_removal": self.max_noise_removal,
            }
        }


# In-memory job store
_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def create_job(
    original_filename: str,
    input_path: str,
    file_size: int = 0,
    breath_remover: bool = False,
    mouth_sounds_remover: bool = False,
    background_music: bool = False,
    helicopter: bool = False,
    water: bool = False,
    dog_barking: bool = False,
    restaurant_chatter: bool = False,
    quality_mode: str = DEFAULT_QUALITY_MODE,
    max_noise_removal: bool = False,
) -> Job:
    """Create a new processing job."""
    # Legacy compat: max_noise_removal=True maps to maximum mode
    if max_noise_removal:
        quality_mode = QualityMode.MAXIMUM

    # Validate quality mode
    valid_modes = {m.value for m in QualityMode}
    if quality_mode not in valid_modes:
        logger.warning(f"Invalid quality_mode '{quality_mode}', using balanced")
        quality_mode = QualityMode.BALANCED

    job_id = str(uuid.uuid4())[:12]
    job = Job(
        job_id=job_id,
        original_filename=original_filename,
        input_path=input_path,
        description=STEP_DESCRIPTIONS[ProcessingStep.QUEUED],
        file_size_bytes=file_size,
        breath_remover=breath_remover,
        mouth_sounds_remover=mouth_sounds_remover,
        background_music=background_music,
        helicopter=helicopter,
        water=water,
        dog_barking=dog_barking,
        restaurant_chatter=restaurant_chatter,
        quality_mode=quality_mode,
        max_noise_removal=max_noise_removal,
    )
    with _lock:
        _jobs[job_id] = job
    logger.info(
        f"Job created: {job_id} for {original_filename} "
        f"(quality_mode={quality_mode})"
    )
    return job


def get_job(job_id: str) -> Optional[Job]:
    """Retrieve a job by ID."""
    with _lock:
        return _jobs.get(job_id)


def update_job_step(job: Job, step: ProcessingStep) -> None:
    """Update job processing step."""
    job.step = step
    job.progress = STEP_PROGRESS[step]
    job.description = STEP_DESCRIPTIONS[step]
    if step == ProcessingStep.COMPLETE:
        job.completed_at = time.time()
    logger.info(f"Job {job.job_id}: {step.value} ({job.progress}%)")


def process_audio(job: Job) -> None:
    """
    Run the full 8-stage audio processing pipeline.
    Quality mode controls aggressiveness at every stage.

    Voice preservation strategy:
    - Each stage uses quality_mode to select gentle/moderate/aggressive params
    - DeepFilterNet uses dry/wet blending to retain vocal texture
    - RNNoise uses mix parameter (0.5 for balanced, skipped for light)
    - Spectral gating: gentle on speech, moderate on silence
    - Adaptive suppression: confidence scaled down by quality mode
    - If SNR is already good, reduce processing intensity further
    """
    from app.services import ffmpeg as ffmpeg_service
    from app.services import deepfilter as deepfilter_service
    from app.services import noise_reduce as noise_reduce_service
    from app.services import rnnoise_service
    from app.services.noise_classifier import classify_noise_from_file
    from app.services.vad import create_speech_mask_from_file
    from app.services.filters import process_custom_filters, apply_adaptive_suppression
    from app.services.quality_scorer import compute_quality_score

    job_dir = TEMP_DIR / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    quality_mode = job.quality_mode
    mode_params = get_mode_params(quality_mode)

    logger.info(
        f"Job {job.job_id}: starting pipeline, quality_mode={quality_mode}, "
        f"deepfilter_atten={mode_params['deepfilter_atten_db']}dB, "
        f"speech_prop={mode_params['speech_prop_decrease']}, "
        f"nonspeech_prop={mode_params['nonspeech_prop_decrease']}"
    )

    try:
        # ============================================================
        # Stage 1: Audio Preprocessing
        # ============================================================
        update_job_step(job, ProcessingStep.PREPROCESSING)

        try:
            info = ffmpeg_service.get_audio_info(job.input_path)
            if "format" in info and "duration" in info["format"]:
                job.duration_seconds = float(info["format"]["duration"])
        except Exception as e:
            logger.warning(f"Could not analyze audio: {e}")

        converted_path = str(job_dir / "converted.wav")
        ffmpeg_service.convert_to_processing_format(job.input_path, converted_path)

        preprocessed_path = str(job_dir / "preprocessed.wav")
        ffmpeg_service.preprocess_audio(
            converted_path, preprocessed_path,
            quality_mode=quality_mode,
        )

        # ============================================================
        # Stage 2: Noise Profiling & Classification
        # ============================================================
        update_job_step(job, ProcessingStep.NOISE_PROFILING)

        noise_profile = classify_noise_from_file(preprocessed_path)
        job.noise_profile = noise_profile.to_dict()

        logger.info(
            f"Job {job.job_id}: SNR={noise_profile.estimated_snr_db:.1f}dB, "
            f"types={noise_profile.detected_types}"
        )

        # Use the requested quality mode directly without auto-downgrading
        effective_mode = quality_mode

        # Generate VAD speech mask
        speech_mask, audio_data, sr = create_speech_mask_from_file(preprocessed_path)

        # ============================================================
        # Stage 3: DeepFilterNet 3 AI Enhancement
        # ============================================================
        update_job_step(job, ProcessingStep.DEEPFILTER)

        denoised_path = str(job_dir / "denoised.wav")
        deepfilter_service.enhance_audio_with_params(
            preprocessed_path,
            denoised_path,
            quality_mode=effective_mode,
        )

        # ============================================================
        # Stage 4: RNNoise Speech Denoising
        # ============================================================
        update_job_step(job, ProcessingStep.RNNOISE)

        rnnoise_path = str(job_dir / "rnnoise.wav")
        rnnoise_service.apply_rnnoise(
            denoised_path, rnnoise_path,
            quality_mode=effective_mode,
        )

        # ============================================================
        # Stage 5: Spectral Gating (VAD-Aware)
        # ============================================================
        update_job_step(job, ProcessingStep.SPECTRAL_GATING)

        gated_path = str(job_dir / "gated.wav")

        # Regenerate speech mask after AI processing
        try:
            speech_mask_post, _, _ = create_speech_mask_from_file(rnnoise_path)
        except Exception:
            speech_mask_post = speech_mask

        noise_reduce_service.reduce_noise_vad_aware(
            rnnoise_path,
            gated_path,
            speech_mask=speech_mask_post,
            noise_spectral_profile=noise_profile.noise_spectral_profile,
            quality_mode=effective_mode,
        )

        # ============================================================
        # Stage 6: Adaptive Noise Suppression
        # ============================================================
        update_job_step(job, ProcessingStep.ADAPTIVE_SUPPRESSION)

        suppressed_path = str(job_dir / "suppressed.wav")

        if noise_profile.detected_types:
            import soundfile as sf
            import numpy as np

            audio_data, sr = sf.read(gated_path, dtype="float32")
            suppressed_audio = apply_adaptive_suppression(
                audio_data, sr, noise_profile.detected_types,
                quality_mode=effective_mode,
            )
            suppressed_audio = np.clip(suppressed_audio, -1.0, 1.0)
            sf.write(suppressed_path, suppressed_audio, sr, subtype="PCM_16")
        else:
            import shutil
            shutil.copy2(gated_path, suppressed_path)

        # Apply custom user-selected filters
        filtered_path = str(job_dir / "filtered.wav")
        process_custom_filters(
            input_path=suppressed_path,
            output_path=filtered_path,
            breath_remover=job.breath_remover,
            mouth_sounds_remover=job.mouth_sounds_remover,
            background_music=job.background_music,
            helicopter=job.helicopter,
            water=job.water,
            dog_barking=job.dog_barking,
            restaurant_chatter=job.restaurant_chatter,
        )

        # ============================================================
        # Stage 7: Loudness Normalization
        # ============================================================
        update_job_step(job, ProcessingStep.NORMALIZING)

        normalized_path = str(job_dir / "normalized.wav")
        ffmpeg_service.normalize_audio(
            filtered_path, normalized_path,
            quality_mode=effective_mode,
        )

        # ============================================================
        # Stage 8: Final Enhancement (Subtle Voice Polish)
        # ============================================================
        update_job_step(job, ProcessingStep.FINAL_ENHANCEMENT)

        enhanced_path = str(job_dir / "final_enhanced.wav")
        ffmpeg_service.apply_final_enhancement(
            normalized_path, enhanced_path,
            quality_mode=quality_mode,
        )

        # ============================================================
        # Export
        # ============================================================
        update_job_step(job, ProcessingStep.EXPORTING)

        output_path = str(job_dir / "output.wav")
        ffmpeg_service.export_to_format(enhanced_path, output_path, "wav")

        mp3_output = str(job_dir / "output.mp3")
        try:
            ffmpeg_service.export_to_format(enhanced_path, mp3_output, "mp3")
        except Exception as e:
            logger.warning(f"MP3 export failed: {e}")

        # ============================================================
        # Quality Scoring
        # ============================================================
        if QUALITY_SCORE_ENABLED:
            try:
                quality_report = compute_quality_score(
                    original_path=converted_path,
                    processed_path=output_path,
                )
                job.quality_report = quality_report.to_dict()
                logger.info(
                    f"Job {job.job_id} quality: "
                    f"improvement={quality_report.improvement_percentage:.0f}%, "
                    f"voice_preservation={quality_report.voice_preservation_score:.0f}, "
                    f"overall={quality_report.overall_quality_score:.0f}"
                )
            except Exception as e:
                logger.warning(f"Quality scoring failed: {e}")

        job.output_path = output_path
        job.original_url_path = job.input_path
        update_job_step(job, ProcessingStep.COMPLETE)

        logger.info(f"Job {job.job_id} completed (mode={quality_mode})")

    except Exception as e:
        logger.error(f"Job {job.job_id} failed: {e}", exc_info=True)
        job.step = ProcessingStep.ERROR
        job.progress = 0
        job.error_message = str(e)
        job.description = f"Error: {str(e)[:200]}"


def start_processing(job: Job) -> None:
    """Start processing a job in a background thread."""
    thread = threading.Thread(
        target=process_audio,
        args=(job,),
        daemon=True,
        name=f"audio-process-{job.job_id}",
    )
    thread.start()
    logger.info(f"Started processing job {job.job_id} (mode={job.quality_mode})")


def cleanup_old_jobs() -> None:
    """Remove expired jobs and their temp files."""
    now = time.time()
    expired_ids = []

    with _lock:
        for job_id, job in _jobs.items():
            if now - job.created_at > FILE_TTL_SECONDS:
                expired_ids.append(job_id)

    for job_id in expired_ids:
        with _lock:
            job = _jobs.pop(job_id, None)
        if job:
            job_dir = TEMP_DIR / job.job_id
            if job_dir.exists():
                import shutil
                shutil.rmtree(job_dir, ignore_errors=True)
            if job.input_path and os.path.exists(job.input_path):
                os.remove(job.input_path)
            logger.info(f"Cleaned up expired job: {job_id}")

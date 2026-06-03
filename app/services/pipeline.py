"""
Audio processing pipeline orchestrator.
Manages the full processing flow from upload to cleaned output.
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

from app.config import TEMP_DIR, FILE_TTL_SECONDS

logger = logging.getLogger(__name__)


class ProcessingStep(str, Enum):
    QUEUED = "queued"
    ANALYZING = "analyzing"
    CONVERTING = "converting"
    DENOISING = "denoising"
    ENHANCING = "enhancing"
    NORMALIZING = "normalizing"
    EXPORTING = "exporting"
    COMPLETE = "complete"
    ERROR = "error"


# Step descriptions for the frontend
STEP_DESCRIPTIONS = {
    ProcessingStep.QUEUED: "Waiting in queue...",
    ProcessingStep.ANALYZING: "Analyzing audio profile...",
    ProcessingStep.CONVERTING: "Preparing audio for processing...",
    ProcessingStep.DENOISING: "AI removing background noise...",
    ProcessingStep.ENHANCING: "Enhancing voice clarity...",
    ProcessingStep.NORMALIZING: "Normalizing audio levels...",
    ProcessingStep.EXPORTING: "Preparing final output...",
    ProcessingStep.COMPLETE: "Processing complete!",
    ProcessingStep.ERROR: "An error occurred",
}

# Progress percentages for each step
STEP_PROGRESS = {
    ProcessingStep.QUEUED: 0,
    ProcessingStep.ANALYZING: 5,
    ProcessingStep.CONVERTING: 15,
    ProcessingStep.DENOISING: 45,
    ProcessingStep.ENHANCING: 70,
    ProcessingStep.NORMALIZING: 85,
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
            "options": {
                "breath_remover": self.breath_remover,
                "mouth_sounds_remover": self.mouth_sounds_remover,
                "background_music": self.background_music,
                "helicopter": self.helicopter,
                "water": self.water,
                "dog_barking": self.dog_barking,
                "restaurant_chatter": self.restaurant_chatter,
            }
        }


# In-memory job store (no database)
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
) -> Job:
    """Create a new processing job with custom options."""
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
    )
    with _lock:
        _jobs[job_id] = job
    logger.info(f"Job created: {job_id} for {original_filename} (Options: breath={breath_remover}, music={background_music})")
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
    Run the full audio processing pipeline.
    This runs in a background thread.

    Pipeline:
    1. Analyze audio metadata
    2. Convert to 48kHz mono WAV
    3. DeepFilterNet AI denoising
    4. Spectral gating for residual noise
    5. Custom target filters (breath, mouth sounds, music, etc.)
    6. Loudness normalization
    7. Export final file
    """
    from app.services import ffmpeg as ffmpeg_service
    from app.services import deepfilter as deepfilter_service
    from app.services import noise_reduce as noise_reduce_service
    from app.services.filters import process_custom_filters

    job_dir = TEMP_DIR / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        # --- Step 1: Analyze ---
        update_job_step(job, ProcessingStep.ANALYZING)
        try:
            info = ffmpeg_service.get_audio_info(job.input_path)
            # Extract duration if available
            if "format" in info and "duration" in info["format"]:
                job.duration_seconds = float(info["format"]["duration"])
        except Exception as e:
            logger.warning(f"Could not analyze audio: {e}")

        # --- Step 2: Convert to processing format ---
        update_job_step(job, ProcessingStep.CONVERTING)
        converted_path = str(job_dir / "converted.wav")
        ffmpeg_service.convert_to_processing_format(job.input_path, converted_path)

        # --- Step 3: DeepFilterNet AI denoising ---
        update_job_step(job, ProcessingStep.DENOISING)
        denoised_path = str(job_dir / "denoised.wav")
        deepfilter_service.enhance_audio(converted_path, denoised_path)

        # --- Step 4: Spectral gating (residual noise) ---
        update_job_step(job, ProcessingStep.ENHANCING)
        enhanced_path = str(job_dir / "enhanced.wav")
        noise_reduce_service.reduce_residual_noise(denoised_path, enhanced_path)

        # --- Custom Filters pass ---
        filtered_path = str(job_dir / "filtered.wav")
        process_custom_filters(
            input_path=enhanced_path,
            output_path=filtered_path,
            breath_remover=job.breath_remover,
            mouth_sounds_remover=job.mouth_sounds_remover,
            background_music=job.background_music,
            helicopter=job.helicopter,
            water=job.water,
            dog_barking=job.dog_barking,
            restaurant_chatter=job.restaurant_chatter,
        )

        # --- Step 5: Loudness normalization ---
        update_job_step(job, ProcessingStep.NORMALIZING)
        normalized_path = str(job_dir / "normalized.wav")
        ffmpeg_service.normalize_audio(filtered_path, normalized_path)

        # --- Step 6: Export ---
        update_job_step(job, ProcessingStep.EXPORTING)
        # Default export to WAV
        output_path = str(job_dir / "output.wav")
        ffmpeg_service.export_to_format(normalized_path, output_path, "wav")

        # Also create MP3 version
        mp3_output = str(job_dir / "output.mp3")
        try:
            ffmpeg_service.export_to_format(normalized_path, mp3_output, "mp3")
        except Exception as e:
            logger.warning(f"MP3 export failed (WAV still available): {e}")

        job.output_path = output_path
        job.original_url_path = job.input_path
        update_job_step(job, ProcessingStep.COMPLETE)

        logger.info(f"Job {job.job_id} completed successfully")

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
    logger.info(f"Started background processing for job {job.job_id}")


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
            # Clean up temp directory
            job_dir = TEMP_DIR / job.job_id
            if job_dir.exists():
                import shutil
                shutil.rmtree(job_dir, ignore_errors=True)
            # Clean up input file
            if job.input_path and os.path.exists(job.input_path):
                os.remove(job.input_path)
            logger.info(f"Cleaned up expired job: {job_id}")

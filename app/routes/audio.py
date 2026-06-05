"""
Audio API routes for upload, status checking, and download.
"""
import os
import logging
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import FileResponse

from app.config import (
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    ALLOWED_EXTENSIONS,
    TEMP_DIR,
    QualityMode,
    DEFAULT_QUALITY_MODE,
)
from app.services.pipeline import (
    create_job,
    get_job,
    start_processing,
    ProcessingStep,
)
from app.services import ffmpeg as ffmpeg_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["audio"])


@router.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
    breath_remover: bool = Form(default=False),
    mouth_sounds_remover: bool = Form(default=False),
    restaurant_chatter: bool = Form(default=False),
    dog_barking: bool = Form(default=False),
    helicopter: bool = Form(default=False),
    background_music: bool = Form(default=False),
    water: bool = Form(default=False),
    quality_mode: str = Form(default=DEFAULT_QUALITY_MODE),
    max_noise_removal: bool = Form(default=False),
):
    """
    Upload an audio file for noise reduction processing.

    Accepts: MP3, WAV, M4A, AAC, FLAC (max 100MB)
    Returns: job_id for status polling

    Quality Modes:
        - light: Minimal noise removal, maximum voice preservation (recommended for clean recordings)
        - balanced: Natural voice with moderate noise removal (DEFAULT)
        - strong: More aggressive noise removal, slight voice impact possible
        - maximum: Maximum suppression, highest artifact risk

    Options:
        - breath_remover: Remove breath sounds
        - mouth_sounds_remover: Remove mouth clicks
        - restaurant_chatter: Suppress background conversations
        - dog_barking: Suppress dog barking
        - helicopter: Remove low-frequency rumble
        - background_music: Attenuate background music
        - water: Remove water/rain hiss
        - max_noise_removal: Legacy flag for maximum mode
    """
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read and validate file size
    contents = await file.read()
    file_size = len(contents)

    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size / 1024 / 1024:.1f}MB). Maximum: {MAX_FILE_SIZE_MB}MB",
        )

    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Save uploaded file to temp directory
    upload_dir = TEMP_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    import uuid
    safe_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    input_path = str(upload_dir / safe_filename)

    with open(input_path, "wb") as f:
        f.write(contents)

    logger.info(
        f"File uploaded: {file.filename} ({file_size / 1024:.1f}KB) -> {input_path} "
        f"(quality_mode={quality_mode})"
    )

    # Create job and start processing
    job = create_job(
        original_filename=file.filename,
        input_path=input_path,
        file_size=file_size,
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

    start_processing(job)

    return {
        "job_id": job.job_id,
        "filename": file.filename,
        "file_size": file_size,
        "message": "Processing started",
        "quality_mode": job.quality_mode,
    }


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Get the current processing status of a job.

    Returns job info including:
    - step: Current processing stage
    - progress: 0-100 percentage
    - quality_report: Quality metrics (when complete)
    - noise_profile: Detected noise types (when available)
    - quality_mode: Processing quality mode used
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job.to_dict()


@router.get("/quality-modes")
async def list_quality_modes():
    """
    List available quality modes with descriptions.
    Frontend can use this to populate a mode selector.
    """
    return {
        "modes": [
            {
                "value": "light",
                "label": "Light",
                "description": "Minimal noise removal, maximum voice preservation. Best for recordings with low background noise.",
                "recommended_for": "Clean recordings, interviews in quiet rooms",
            },
            {
                "value": "balanced",
                "label": "Balanced (Recommended)",
                "description": "Natural voice quality with moderate noise removal. Default mode. Preserves speech dynamics and tone.",
                "recommended_for": "Most recordings, podcasts, voiceovers",
                "default": True,
            },
            {
                "value": "pure_voice",
                "label": "Pure Voice",
                "description": "100% original voice preservation. Bypasses all EQ, coloring, and compression, using AI to pull out background noise only.",
                "recommended_for": "Recordings where you want zero changes to your natural voice tone and dynamics",
            },
            {
                "value": "social_media",
                "label": "Social Media Ready",
                "description": "Optimized for mobile speakers. Delivers maximum vocal clarity, warmth, and presence with -14 LUFS level target.",
                "recommended_for": "YouTube, TikTok, Reels, Shorts, and Podcasts",
            },
        ],
        "default": "balanced",
    }


@router.get("/download/{job_id}")
async def download_audio(
    job_id: str,
    format: str = Query(default="wav", regex="^(wav|mp3)$"),
):
    """Download the processed (cleaned) audio file."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.step == ProcessingStep.ERROR:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {job.error_message}",
        )

    if job.step != ProcessingStep.COMPLETE:
        raise HTTPException(
            status_code=202,
            detail=f"Processing not yet complete. Current step: {job.step.value}",
        )

    job_dir = TEMP_DIR / job.job_id
    if format == "mp3":
        output_file = str(job_dir / "output.mp3")
        media_type = "audio/mpeg"
    else:
        output_file = str(job_dir / "output.wav")
        media_type = "audio/wav"

    if not os.path.exists(output_file):
        alt_format = "wav" if format == "mp3" else "mp3"
        alt_file = str(job_dir / f"output.{alt_format}")
        if os.path.exists(alt_file):
            output_file = alt_file
            media_type = "audio/wav" if alt_format == "wav" else "audio/mpeg"
        else:
            raise HTTPException(status_code=404, detail="Output file not found")

    original_stem = Path(job.original_filename).stem
    download_filename = f"{original_stem}_cleaned.{format}"

    return FileResponse(
        path=output_file,
        media_type=media_type,
        filename=download_filename,
        headers={
            "Content-Disposition": f'attachment; filename="{download_filename}"'
        },
    )


@router.get("/download-original/{job_id}")
async def download_original(job_id: str):
    """Serve the original uploaded audio for comparison playback."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not os.path.exists(job.input_path):
        raise HTTPException(status_code=404, detail="Original file not found")

    ext = Path(job.original_filename).suffix.lower()
    media_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".flac": "audio/flac",
    }

    return FileResponse(
        path=job.input_path,
        media_type=media_types.get(ext, "audio/wav"),
        filename=job.original_filename,
    )

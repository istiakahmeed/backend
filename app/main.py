"""
FastAPI application entry point.
AI Audio Noise Cancellation Tool - Backend Server.
"""
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS, TEMP_DIR
from app.routes.audio import router as audio_router
from app.services.pipeline import cleanup_old_jobs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def periodic_cleanup():
    """Background task to clean up expired jobs every 10 minutes."""
    while True:
        await asyncio.sleep(600)  # 10 minutes
        try:
            cleanup_old_jobs()
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    # --- Startup ---
    logger.info("=" * 60)
    logger.info("  AI Audio Noise Cancellation Tool - Starting")
    logger.info("=" * 60)

    # Ensure temp directory exists
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Temp directory: {TEMP_DIR}")

    # Pre-load DeepFilterNet model
    logger.info("Loading AI models...")
    try:
        from app.services.deepfilter import init_model
        model_loaded = init_model()
        if model_loaded:
            logger.info("✓ DeepFilterNet model loaded successfully")
        else:
            logger.warning(
                "✗ DeepFilterNet model could not be loaded. "
                "Using fallback noise reduction methods only."
            )
    except Exception as e:
        logger.error(f"Error during model initialization: {e}")
        logger.warning("Server starting with fallback noise reduction only.")

    # Start cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    logger.info("Periodic cleanup task started")

    logger.info("=" * 60)
    logger.info("  Server ready! Accepting requests.")
    logger.info("=" * 60)

    yield

    # --- Shutdown ---
    logger.info("Shutting down...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("Cleanup task stopped. Goodbye.")


# Create FastAPI app
app = FastAPI(
    title="AI Audio Noise Cancellation",
    description="Remove background noise from audio files using DeepFilterNet 3 AI.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(audio_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from app.services.deepfilter import get_model_info
    return {
        "status": "healthy",
        "service": "AI Audio Noise Cancellation",
        "model": get_model_info(),
    }

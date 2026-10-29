import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response

from utils import TranscriptionService
from routers import transcribe_router, analytics_router


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(CURRENT_DIR, "utils")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lifecycle Context: Instantiate engine orchestrator with cluster attributes
    service = TranscriptionService(config_dir=CONFIG_DIR)
    await service.initialize()
    app.state.transcription_service = service
    yield
    # Teardown Hook Context: Clean cluster cache pipelines, models, and references
    await service.close(timeout_s=3.0)

app = FastAPI(
    title="High-Concurrency Whisper Backend Stack",
    version="1.0.0",
    lifespan=lifespan
)

# Register clean isolated feature packages mapping functional domains
app.include_router(transcribe_router)
app.include_router(analytics_router)

@app.get("/health")
async def check_health():
    """Check if the dependent services are healthy"""
    return Response(status_code=200)
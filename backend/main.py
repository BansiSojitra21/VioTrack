"""TrafficAI FastAPI entrypoint."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import preload_frame_engine, router
from app.core.config import settings
from app.core.database import init_db
from app.core.logging import configure_logging
from app.services import repository


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    settings.resolved_data_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    interrupted_jobs = repository.fail_interrupted_jobs()
    if interrupted_jobs:
        logging.getLogger(__name__).warning(
            "marked %s interrupted analysis job(s) as failed", interrupted_jobs
        )
    preload_error = preload_frame_engine()
    if preload_error:
        logging.getLogger(__name__).warning(
            "Could not preload the live vehicle model: %s", preload_error
        )
    yield


app = FastAPI(
    title="TrafficAI Inference API",
    version=__version__,
    summary="YOLO traffic detection, tracking, and potential-violation analysis.",
    description=(
        "Separate inference service for TrafficAI. "
        "Heavy YOLO inference stays here so the Next.js app can deploy on Vercel. "
        "Potential violations are evidence for review, not automated penalties."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root() -> dict:
    return {"service": "TrafficAI", "docs": "/docs", "health": "/api/health"}

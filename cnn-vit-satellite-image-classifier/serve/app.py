"""FastAPI inference server for the satellite image classifier.

Endpoints
---------
GET  /health          Server health and loaded-model summary.
GET  /models          List all four models with their load status.
POST /predict         Classify a satellite image tile.

Usage
-----
    uvicorn serve.app:app --host 0.0.0.0 --port 8000

    # or via Docker Compose (see docker-compose.yml)
    docker compose up

Example predict call
--------------------
    curl -X POST "http://localhost:8000/predict?model=pytorch_vit" \\
         -F "file=@tile.jpg"
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from serve.model_registry import ModelRegistry, _BACKEND, _MODEL_FILES
from serve.schemas import (
    DEFAULT_MODEL,
    HealthResponse,
    ModelName,
    ModelStatus,
    PredictResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Resolve model directory: env var → project default
MODEL_DIR = Path(
    os.getenv(
        "MODEL_DIR",
        str(Path(__file__).parents[1] / "models" / "trained"),
    )
)

_registry = ModelRegistry(MODEL_DIR)
_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once at startup; release nothing (models stay hot)."""
    _registry.load_all()
    yield


app = FastAPI(
    title="Satellite Image Classifier",
    description=(
        "Inference server for agricultural vs non-agricultural land classification. "
        "Serves four model variants — Keras CNN, Keras CNN-ViT, PyTorch CNN, "
        "and PyTorch CNN-ViT — selectable per request."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Server health and loaded-model summary",
)
def health() -> HealthResponse:
    loaded = _registry.loaded_names
    return HealthResponse(
        status="ok" if loaded else "degraded",
        models_loaded=loaded,
        models_unavailable=_registry.unavailable_names,
        uptime_s=round(time.time() - _start_time, 1),
    )


@app.get(
    "/models",
    response_model=list[ModelStatus],
    summary="List all models with their availability",
)
def list_models() -> list[ModelStatus]:
    return [
        ModelStatus(
            name=name,
            backend=_BACKEND[name],
            loaded=_registry.is_loaded(name),
        )
        for name in _MODEL_FILES
    ]


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify a satellite image tile",
)
async def predict(
    file: Annotated[
        UploadFile,
        File(description="RGB satellite image tile (JPEG or PNG)"),
    ],
    model: Annotated[
        ModelName,
        Query(description="Model variant to use for inference"),
    ] = DEFAULT_MODEL,
) -> PredictResponse:
    if not _registry.is_loaded(model):
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model}' is not loaded. "
                   f"Loaded models: {_registry.loaded_names}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    t0 = time.perf_counter()
    try:
        prediction, confidence, class_id = _registry.predict(model, image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return PredictResponse(
        model=model,
        prediction=prediction,
        confidence=confidence,
        class_id=class_id,
        latency_ms=latency_ms,
    )

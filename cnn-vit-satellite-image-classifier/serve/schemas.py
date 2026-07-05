"""Pydantic request / response schemas for the inference API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ModelName = Literal["keras_cnn", "keras_vit", "pytorch_cnn", "pytorch_vit"]

DEFAULT_MODEL: ModelName = "pytorch_vit"


class PredictResponse(BaseModel):
    model: str
    prediction: str
    confidence: float
    class_id: int
    latency_ms: float


class ModelStatus(BaseModel):
    name: str
    backend: Literal["keras", "pytorch"]
    loaded: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models_loaded: list[str]
    models_unavailable: list[str]
    uptime_s: float

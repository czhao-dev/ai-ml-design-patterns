"""Tests for serve/schemas.py."""

import pytest
from pydantic import ValidationError

from serve.schemas import DEFAULT_MODEL, HealthResponse, ModelStatus, PredictResponse


def test_default_model_is_a_valid_model_name():
    PredictResponse(
        model=DEFAULT_MODEL, prediction="agricultural", confidence=0.9, class_id=1, latency_ms=12.3
    )


def test_predict_response_requires_all_fields():
    with pytest.raises(ValidationError):
        PredictResponse(model="pytorch_vit", prediction="agricultural")


def test_model_status_rejects_unknown_backend():
    with pytest.raises(ValidationError):
        ModelStatus(name="pytorch_vit", backend="onnx", loaded=True)


def test_health_response_rejects_unknown_status():
    with pytest.raises(ValidationError):
        HealthResponse(status="fine", models_loaded=[], models_unavailable=[], uptime_s=1.0)


def test_health_response_accepts_ok_and_degraded():
    ok = HealthResponse(status="ok", models_loaded=["pytorch_vit"], models_unavailable=[], uptime_s=1.0)
    degraded = HealthResponse(
        status="degraded", models_loaded=[], models_unavailable=["keras_cnn"], uptime_s=1.0
    )
    assert ok.status == "ok"
    assert degraded.status == "degraded"

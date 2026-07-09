"""Tests for serve/model_registry.py.

These tests inject hand-written fake models directly into the registry's
internal cache instead of loading real checkpoints, so they exercise the
prediction/status logic without needing model files or a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from serve.model_registry import CLASS_NAMES, ModelRegistry


class FakeKerasModel:
    """Mimics `tf.keras.Model.predict`, scripted with a fixed output."""

    def __init__(self, output: np.ndarray) -> None:
        self._output = output

    def predict(self, arr, verbose=0):
        return self._output


class FakePyTorchModel:
    """Mimics a `torch.nn.Module`'s `__call__`, scripted with fixed logits."""

    def __init__(self, logits: torch.Tensor) -> None:
        self._logits = logits

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._logits


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path)


def test_predict_raises_for_unloaded_model(registry, rgb_png_bytes):
    with pytest.raises(ValueError, match="pytorch_vit"):
        registry.predict("pytorch_vit", rgb_png_bytes())


def test_predict_keras_sigmoid_output_above_threshold_is_agricultural(registry, rgb_png_bytes):
    registry._models["keras_cnn"] = FakeKerasModel(np.array([[0.9]], dtype=np.float32))

    label, confidence, class_id = registry.predict("keras_cnn", rgb_png_bytes())

    assert class_id == 1
    assert label == CLASS_NAMES[1]
    assert confidence == pytest.approx(0.9)


def test_predict_keras_sigmoid_output_below_threshold_is_non_agricultural(registry, rgb_png_bytes):
    registry._models["keras_cnn"] = FakeKerasModel(np.array([[0.2]], dtype=np.float32))

    label, confidence, class_id = registry.predict("keras_cnn", rgb_png_bytes())

    assert class_id == 0
    assert label == CLASS_NAMES[0]
    assert confidence == pytest.approx(0.8)


def test_predict_keras_softmax_output_picks_argmax_class(registry, rgb_png_bytes):
    registry._models["keras_vit"] = FakeKerasModel(np.array([[0.1, 0.9]], dtype=np.float32))

    label, confidence, class_id = registry.predict("keras_vit", rgb_png_bytes())

    assert class_id == 1
    assert label == CLASS_NAMES[1]
    assert confidence == pytest.approx(0.9)


def test_predict_pytorch_uses_softmax_argmax(registry, rgb_png_bytes):
    logits = torch.tensor([[5.0, 1.0]])
    registry._models["pytorch_cnn"] = FakePyTorchModel(logits)

    label, confidence, class_id = registry.predict("pytorch_cnn", rgb_png_bytes())

    assert class_id == 0
    assert label == CLASS_NAMES[0]
    assert 0.0 < confidence <= 1.0


def test_loaded_and_unavailable_names_reflect_registered_models(registry):
    registry._models["pytorch_cnn"] = FakePyTorchModel(torch.tensor([[1.0, 0.0]]))

    assert registry.loaded_names == ["pytorch_cnn"]
    assert registry.unavailable_names == ["keras_cnn", "keras_vit", "pytorch_vit"]
    assert registry.is_loaded("pytorch_cnn") is True
    assert registry.is_loaded("keras_cnn") is False


def test_backend_lookup_by_model_name(registry):
    assert registry.backend("keras_cnn") == "keras"
    assert registry.backend("pytorch_vit") == "pytorch"

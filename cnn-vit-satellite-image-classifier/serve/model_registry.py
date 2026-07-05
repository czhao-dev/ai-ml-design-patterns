"""Model registry — loads and caches all four classifiers at startup.

Supported models
----------------
keras_cnn   Keras CNN baseline          (sigmoid binary output)
keras_vit   Keras CNN-ViT hybrid        (softmax 2-class output)
pytorch_cnn PyTorch CNN baseline        (CrossEntropy logits, 2-class)
pytorch_vit PyTorch CNN-ViT hybrid      (CrossEntropy logits, 2-class)

A missing model file causes a warning, not a crash: the model is simply
omitted from the registry and the /health endpoint reports it as
unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from serve.preprocessing import preprocess_for_keras, preprocess_for_pytorch
from serve.pytorch_models import CNN_ViT_Hybrid, build_satellite_cnn

logger = logging.getLogger(__name__)

# Human-readable class labels (ImageFolder alphabetical order)
CLASS_NAMES: dict[int, str] = {0: "non-agricultural", 1: "agricultural"}

_MODEL_FILES: dict[str, str] = {
    "keras_cnn": "ai_capstone_keras_best_model.model.keras",
    "keras_vit": "keras_cnn_vit_ai_capstone.keras",
    "pytorch_cnn": "ai_capstone_pytorch_state_dict.pth",
    "pytorch_vit": "pytorch_cnn_vit_ai_capstone_model_state_dict.pth",
}

_BACKEND: dict[str, str] = {
    "keras_cnn": "keras",
    "keras_vit": "keras",
    "pytorch_cnn": "pytorch",
    "pytorch_vit": "pytorch",
}


class ModelRegistry:
    """Thread-safe (single-process) container for all loaded models."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        self._models: dict[str, Any] = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self._device)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Load every model. Missing files are skipped with a warning."""
        self._load_keras_models()
        self._load_pytorch_models()
        logger.info(
            "Registry ready. Loaded: %s | Unavailable: %s",
            self.loaded_names,
            self.unavailable_names,
        )

    def _load_keras_models(self) -> None:
        try:
            import tensorflow as tf
            import serve.keras_custom_layers  # noqa: F401 — registers custom layers
        except ImportError:
            logger.warning("TensorFlow not available; Keras models will not be loaded.")
            return

        for name in ("keras_cnn", "keras_vit"):
            path = self.model_dir / _MODEL_FILES[name]
            if not path.exists():
                logger.warning("Model file not found, skipping: %s", path)
                continue
            try:
                self._models[name] = tf.keras.models.load_model(str(path))
                logger.info("Loaded %s", name)
            except Exception:
                logger.exception("Failed to load %s", name)

    def _load_pytorch_models(self) -> None:
        # Standalone CNN
        self._load_pytorch(
            name="pytorch_cnn",
            model=build_satellite_cnn(num_classes=2),
            strict=True,
            key_prefix=None,
        )
        # CNN-ViT hybrid (CNN backbone weights saved under "cnn.*" prefix)
        self._load_pytorch(
            name="pytorch_vit",
            model=CNN_ViT_Hybrid(num_classes=2),
            strict=False,
            key_prefix="cnn.",
        )

    def _load_pytorch(
        self,
        name: str,
        model: torch.nn.Module,
        strict: bool,
        key_prefix: str | None,
    ) -> None:
        path = self.model_dir / _MODEL_FILES[name]
        if not path.exists():
            logger.warning("Model file not found, skipping: %s", path)
            return
        try:
            state_dict = torch.load(str(path), map_location=self._device)
            # Unwrap common checkpoint wrappers
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            if not strict and key_prefix is not None:
                # Strip "module." and the given prefix so keys align with the
                # model's state dict (used for the CNN-ViT hybrid backbone).
                cleaned: dict[str, torch.Tensor] = {}
                model_keys = set(model.state_dict())
                for k, v in state_dict.items():
                    k = k.removeprefix("module.").removeprefix(key_prefix)
                    if k in model_keys and model.state_dict()[k].shape == v.shape:
                        cleaned[k] = v
                missing, unexpected = model.load_state_dict(cleaned, strict=False)
                if missing:
                    logger.debug("%s: %d tensors at init defaults", name, len(missing))
                if unexpected:
                    logger.debug("%s: %d unexpected keys ignored", name, len(unexpected))
            else:
                model.load_state_dict(state_dict, strict=strict)

            model.to(self._device).eval()
            self._models[name] = model
            logger.info("Loaded %s", name)
        except Exception:
            logger.exception("Failed to load %s", name)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, name: str, image_bytes: bytes) -> tuple[str, float, int]:
        """Return ``(label, confidence, class_id)`` for a raw image.

        Raises ``ValueError`` if *name* is not loaded.
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' is not loaded.")
        if _BACKEND[name] == "keras":
            return self._predict_keras(name, image_bytes)
        return self._predict_pytorch(name, image_bytes)

    def _predict_keras(self, name: str, image_bytes: bytes) -> tuple[str, float, int]:
        arr = preprocess_for_keras(image_bytes)
        pred = self._models[name].predict(arr, verbose=0)

        if pred.shape[-1] == 1:
            # keras_cnn: sigmoid scalar → binary threshold
            prob_agri = float(pred[0][0])
            class_id = 1 if prob_agri > 0.5 else 0
            confidence = prob_agri if class_id == 1 else 1.0 - prob_agri
        else:
            # keras_vit: softmax over 2 classes
            probs = pred[0]
            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])

        return CLASS_NAMES[class_id], round(confidence, 4), class_id

    def _predict_pytorch(self, name: str, image_bytes: bytes) -> tuple[str, float, int]:
        tensor = preprocess_for_pytorch(image_bytes).to(self._device)
        with torch.no_grad():
            logits = self._models[name](tensor)
            probs = F.softmax(logits, dim=1)[0]
        class_id = int(probs.argmax().item())
        confidence = round(float(probs[class_id].item()), 4)
        return CLASS_NAMES[class_id], confidence, class_id

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    @property
    def loaded_names(self) -> list[str]:
        return sorted(self._models)

    @property
    def unavailable_names(self) -> list[str]:
        return sorted(k for k in _MODEL_FILES if k not in self._models)

    def is_loaded(self, name: str) -> bool:
        return name in self._models

    def backend(self, name: str) -> str:
        return _BACKEND[name]

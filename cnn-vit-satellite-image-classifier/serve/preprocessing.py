"""Image preprocessing — one pipeline per model backend.

Keras models were trained with pixel values rescaled to [0, 1].
PyTorch models were trained with ImageNet mean/std normalisation.
Both expect 64×64 RGB input.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

IMG_SIZE = 64


# ---------------------------------------------------------------------------
# Keras preprocessing
# ---------------------------------------------------------------------------

def preprocess_for_keras(image_bytes: bytes) -> "np.ndarray":
    """Return a (1, 64, 64, 3) float32 array with values in [0, 1]."""
    img = (
        Image.open(io.BytesIO(image_bytes))
        .convert("RGB")
        .resize((IMG_SIZE, IMG_SIZE))
    )
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis]  # (1, H, W, 3)


# ---------------------------------------------------------------------------
# PyTorch preprocessing
# ---------------------------------------------------------------------------

def preprocess_for_pytorch(image_bytes: bytes) -> "torch.Tensor":
    """Return a (1, 3, 64, 64) tensor with ImageNet normalisation."""
    import torch
    from torchvision import transforms

    _transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _transform(img).unsqueeze(0)  # (1, C, H, W)

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def rgb_png_bytes():
    """A small in-memory RGB PNG, standing in for an uploaded satellite tile."""

    def _make(size: tuple[int, int] = (32, 32), color: tuple[int, int, int] = (10, 120, 200)) -> bytes:
        image = Image.fromarray(np.full((size[1], size[0], 3), color, dtype=np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    return _make

"""Tests for serve/preprocessing.py."""

import numpy as np
import torch

from serve.preprocessing import IMG_SIZE, preprocess_for_keras, preprocess_for_pytorch


def test_preprocess_for_keras_shape_and_range(rgb_png_bytes):
    arr = preprocess_for_keras(rgb_png_bytes())

    assert arr.shape == (1, IMG_SIZE, IMG_SIZE, 3)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


def test_preprocess_for_keras_rescales_pixel_values(rgb_png_bytes):
    arr = preprocess_for_keras(rgb_png_bytes(color=(255, 0, 0)))
    np.testing.assert_allclose(arr[0, 0, 0], [1.0, 0.0, 0.0], atol=1e-6)


def test_preprocess_for_keras_resizes_non_square_input(rgb_png_bytes):
    arr = preprocess_for_keras(rgb_png_bytes(size=(128, 64)))
    assert arr.shape == (1, IMG_SIZE, IMG_SIZE, 3)


def test_preprocess_for_pytorch_shape_and_dtype(rgb_png_bytes):
    tensor = preprocess_for_pytorch(rgb_png_bytes())

    assert tensor.shape == (1, 3, IMG_SIZE, IMG_SIZE)
    assert tensor.dtype == torch.float32


def test_preprocess_for_pytorch_is_not_left_in_zero_to_one_range(rgb_png_bytes):
    # ImageNet normalisation should shift/scale pixel values away from [0, 1]
    # for a saturated color channel.
    tensor = preprocess_for_pytorch(rgb_png_bytes(color=(255, 255, 255)))
    assert tensor.max().item() > 1.0

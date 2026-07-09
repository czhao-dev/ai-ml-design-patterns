"""Tests for src/student_model.py."""

import torch

from src.student_model import StudentCNN


def test_forward_shape_default_classes():
    torch.manual_seed(0)
    model = StudentCNN().eval()
    logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 2)


def test_forward_respects_num_classes():
    model = StudentCNN(num_classes=5).eval()
    logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 5)


def test_param_count_is_near_the_documented_259k():
    model = StudentCNN()
    n_params = sum(p.numel() for p in model.parameters())
    assert 200_000 < n_params < 320_000


def test_is_meaningfully_smaller_than_the_teacher_cnn():
    from src.paths import build_satellite_cnn

    student_params = sum(p.numel() for p in StudentCNN().parameters())
    teacher_params = sum(p.numel() for p in build_satellite_cnn().parameters())

    assert student_params < teacher_params / 10

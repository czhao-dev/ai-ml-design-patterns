"""Tests for src/eval_utils.py.

`build_canonical_split` needs the (gitignored, downloaded-on-first-run)
satellite image dataset and `load_fp32_cnn`/`load_fp32_cnn_vit` need the
(gitignored) trained checkpoints from the sibling project, so both are out
of scope for these offline unit tests. `evaluate_model` is pure orchestration
over a model and a loader, so it's exercised here with a fake model and a
small in-memory "loader".
"""

from __future__ import annotations

import torch

from src.eval_utils import evaluate_model


class _PerfectClassifier(torch.nn.Module):
    """Outputs high-confidence logits for the true label of each input."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images encode the intended label in their first pixel: 0.0 -> class
        # 0, 1.0 -> class 1. Real models obviously don't do this; it's just a
        # deterministic stand-in so we know the "correct" prediction.
        labels = images[:, 0, 0, 0].long()
        logits = torch.zeros(images.shape[0], 2)
        logits[torch.arange(images.shape[0]), labels] = 10.0
        return logits


def _make_batch(labels: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.zeros(len(labels), 3, 4, 4)
    for i, label in enumerate(labels):
        images[i, 0, 0, 0] = float(label)
    return images, torch.tensor(labels)


def test_evaluate_model_perfect_classifier_scores_one_on_every_metric():
    loader = [_make_batch([0, 1, 0, 1])]
    model = _PerfectClassifier()

    metrics = evaluate_model(model, loader, device="cpu")

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_evaluate_model_aggregates_across_multiple_batches():
    loader = [_make_batch([0, 1]), _make_batch([1, 0])]
    model = _PerfectClassifier()

    metrics = evaluate_model(model, loader, device="cpu")

    assert metrics["accuracy"] == 1.0

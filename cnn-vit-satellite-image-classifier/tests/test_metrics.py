"""Tests for src/metrics.py."""

import numpy as np
import pytest

from src.metrics import binary_classification_metrics


def test_perfect_predictions_score_one_on_every_metric():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.05, 0.1, 0.9, 0.95])

    metrics = binary_classification_metrics(y_true, y_score)

    assert metrics == pytest.approx(
        {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "roc_auc": 1.0}
    )


def test_inverted_predictions_score_zero_precision_and_recall():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.95, 0.9, 0.1, 0.05])

    metrics = binary_classification_metrics(y_true, y_score)

    assert metrics["accuracy"] == pytest.approx(0.0)
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["roc_auc"] == pytest.approx(0.0)


def test_threshold_controls_the_decision_boundary():
    y_true = np.array([0, 1])
    y_score = np.array([0.5, 0.5])

    below_threshold = binary_classification_metrics(y_true, y_score, threshold=0.6)
    at_threshold = binary_classification_metrics(y_true, y_score, threshold=0.5)

    # threshold=0.6 -> both scores fall below it -> both predicted class 0
    assert below_threshold["accuracy"] == pytest.approx(0.5)
    # threshold=0.5 -> both scores meet it -> both predicted class 1
    assert at_threshold["accuracy"] == pytest.approx(0.5)


def test_accepts_plain_python_lists():
    metrics = binary_classification_metrics([0, 1, 1], [0.1, 0.6, 0.9])
    assert metrics["accuracy"] == pytest.approx(1.0)

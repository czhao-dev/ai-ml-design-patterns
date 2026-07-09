"""Tests for src/visualization.py."""

import matplotlib

matplotlib.use("Agg")

from src.visualization import plot_confusion_matrix


def test_plot_confusion_matrix_uses_given_labels_and_title():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]

    ax = plot_confusion_matrix(y_true, y_pred, labels=["non-agri", "agri"], title="Test CM")

    assert ax.get_title() == "Test CM"
    assert ax.get_xlabel() == "Predicted label"
    assert ax.get_ylabel() == "True label"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["non-agri", "agri"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["non-agri", "agri"]


def test_plot_confusion_matrix_default_title():
    ax = plot_confusion_matrix([0, 1], [0, 1], labels=["a", "b"])
    assert ax.get_title() == "Confusion Matrix"

"""Plotting helpers for reports and analysis scripts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels: list[str], title: str = "Confusion Matrix"):
    """Plot a labeled confusion matrix and return the Matplotlib axis."""
    matrix = confusion_matrix(y_true, y_pred)
    _, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    return ax

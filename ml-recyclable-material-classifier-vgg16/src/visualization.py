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


def plot_training_curves(history: dict, title_prefix: str, save_path_prefix=None):
    """Plot and optionally save loss and accuracy curves from a Keras History.history dict."""
    plt.figure(figsize=(5, 5))
    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title(f"{title_prefix} - Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    if save_path_prefix is not None:
        plt.savefig(f"{save_path_prefix}_loss.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(history["accuracy"], label="Training Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{title_prefix} - Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    if save_path_prefix is not None:
        plt.savefig(f"{save_path_prefix}_accuracy.png", bbox_inches="tight")
    plt.close()


def plot_sample_predictions(images, actual_labels, predicted_labels, model_name: str, save_path=None, n_samples: int = 2):
    """Plot a few sample images with actual vs. predicted labels, and optionally save the figure."""
    n_samples = min(n_samples, len(images))
    fig, axes = plt.subplots(1, n_samples, figsize=(5 * n_samples, 5))
    if n_samples == 1:
        axes = [axes]

    for ax, image, actual, predicted in zip(axes, images, actual_labels, predicted_labels):
        ax.imshow(image.astype("uint8"))
        ax.set_title(f"{model_name}\nActual: {actual}, Predicted: {predicted}")
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()

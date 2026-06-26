#!/usr/bin/env python3
"""Evaluate the feature-extraction and fine-tuned models on the full held-out test set."""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import tensorflow as tf
from sklearn import metrics as sk_metrics

from src.config import (
    FEATURE_EXTRACTION_MODEL_PATH,
    FIGURES_DIR,
    FINE_TUNED_MODEL_PATH,
    IMG_SIZE,
    REPORTS_DIR,
    TEST_DIR,
)
from src.data_utils import list_image_files
from src.metrics import binary_classification_metrics
from src.visualization import plot_confusion_matrix, plot_sample_predictions

CLASS_TO_LABEL = {"crack": 0, "dent": 1}
LABEL_TO_CLASS = {0: "crack", 1: "dent"}


def load_test_set():
    test_files = list_image_files(TEST_DIR)
    images = np.array(
        [
            tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(f, target_size=IMG_SIZE)
            )
            for f in test_files
        ]
    )
    labels = np.array([CLASS_TO_LABEL[f.parent.name] for f in test_files])
    images_scaled = images.astype("float32") / 255.0
    return test_files, images, images_scaled, labels


def evaluate_model(model_path, model_name, images, images_scaled, labels):
    model = tf.keras.models.load_model(model_path)
    scores = model.predict(images_scaled, verbose=0).flatten()
    predictions = (scores >= 0.5).astype(int)

    print(f"\n{model_name}")
    print(sk_metrics.classification_report(labels, predictions, target_names=["crack", "dent"]))

    computed_metrics = binary_classification_metrics(labels, scores)
    for key, value in computed_metrics.items():
        print(f"  {key}: {value:.4f}")

    return scores, predictions, computed_metrics


def main() -> None:
    test_files, images, images_scaled, labels = load_test_set()
    print(f"Loaded {len(test_files)} test images: crack={np.sum(labels == 0)}, dent={np.sum(labels == 1)}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for model_path, model_name, fig_prefix in [
        (FEATURE_EXTRACTION_MODEL_PATH, "Feature-Extraction Model", "feature_extraction"),
        (FINE_TUNED_MODEL_PATH, "Fine-Tuned Model", "fine_tuned"),
    ]:
        scores, predictions, computed_metrics = evaluate_model(model_path, model_name, images, images_scaled, labels)
        results[model_name] = computed_metrics

        plot_confusion_matrix(
            labels, predictions, labels=["crack", "dent"], title=f"{model_name} - Confusion Matrix"
        )
        import matplotlib.pyplot as plt

        plt.savefig(FIGURES_DIR / f"{fig_prefix}_confusion_matrix.png", bbox_inches="tight")
        plt.close()

        sample_indices = [0, 1]
        plot_sample_predictions(
            images=images[sample_indices],
            actual_labels=[LABEL_TO_CLASS[label] for label in labels[sample_indices]],
            predicted_labels=[LABEL_TO_CLASS[pred] for pred in predictions[sample_indices]],
            model_name=model_name,
            save_path=FIGURES_DIR / f"{fig_prefix}_sample_predictions.png",
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {REPORTS_DIR / 'evaluation_metrics.json'}")


if __name__ == "__main__":
    main()

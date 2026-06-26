#!/usr/bin/env python3
"""Download the dataset and build the train/validation/test image generators."""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import (
    BATCH_SIZE,
    DATASET_ARCHIVE,
    DATASET_URL,
    EXTRACTED_DATASET_DIR,
    FIGURES_DIR,
    IMG_SIZE,
    RAW_DATA_DIR,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
    VAL_SPLIT,
)
from src.data_utils import extract_zip_archive, summarize_class_distribution


def download_dataset() -> None:
    if DATASET_ARCHIVE.exists():
        print(f"Dataset archive already downloaded at {DATASET_ARCHIVE}")
        return

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {DATASET_URL}")
    with requests.get(DATASET_URL, stream=True) as response:
        response.raise_for_status()
        with open(DATASET_ARCHIVE, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Saved archive to {DATASET_ARCHIVE}")


def extract_dataset() -> None:
    if EXTRACTED_DATASET_DIR.exists():
        print(f"Dataset already extracted at {EXTRACTED_DATASET_DIR}")
        return

    print("Extracting dataset archive...")
    extract_zip_archive(DATASET_ARCHIVE, RAW_DATA_DIR)
    print(f"Extracted dataset to {EXTRACTED_DATASET_DIR}")


def build_generators():
    train_datagen = ImageDataGenerator(
        validation_split=VAL_SPLIT,
        rescale=1.0 / 255.0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    val_datagen = ImageDataGenerator(
        validation_split=VAL_SPLIT,
        rescale=1.0 / 255.0,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        seed=SEED,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        target_size=IMG_SIZE,
        subset="training",
    )
    val_generator = val_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        seed=SEED,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        target_size=IMG_SIZE,
        subset="validation",
    )
    test_generator = test_datagen.flow_from_directory(
        directory=TEST_DIR,
        seed=SEED,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
        target_size=IMG_SIZE,
    )

    return train_datagen, train_generator, val_generator, test_generator


def plot_augmented_samples(train_datagen, save_path) -> None:
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt

    sample_files = sorted((TRAIN_DIR / "O").glob("*"))[:1]
    sample_img = tf.keras.preprocessing.image.img_to_array(
        tf.keras.preprocessing.image.load_img(sample_files[0], target_size=(100, 100))
    )
    sample_img = np.expand_dims(sample_img, axis=0)
    sample_label = np.array(["O"])

    sample_generator = train_datagen.flow(sample_img, sample_label, batch_size=1)
    samples = [next(sample_generator) for _ in range(5)]

    fig, axes = plt.subplots(1, 5, figsize=(16, 6))
    for ax, (image_batch, label_batch) in zip(axes, samples):
        ax.imshow(image_batch[0])
        ax.set_title(f"Label: {label_batch[0]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved augmented sample figure to {save_path}")


def main() -> None:
    download_dataset()
    extract_dataset()

    print("\nClass distribution:")
    print("  train:", summarize_class_distribution(TRAIN_DIR))
    print("  test: ", summarize_class_distribution(TEST_DIR))

    train_datagen, train_generator, val_generator, test_generator = build_generators()

    print("\nGenerator summary:")
    print(f"  train_generator: {train_generator.samples} images, class_indices={train_generator.class_indices}, batches={len(train_generator)}")
    print(f"  val_generator:   {val_generator.samples} images, batches={len(val_generator)}")
    print(f"  test_generator:  {test_generator.samples} images, batches={len(test_generator)}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_augmented_samples(train_datagen, FIGURES_DIR / "augmented_samples.png")


if __name__ == "__main__":
    main()

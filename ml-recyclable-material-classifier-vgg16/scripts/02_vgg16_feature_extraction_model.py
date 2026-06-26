#!/usr/bin/env python3
"""Train a VGG16 feature-extraction model: frozen convolutional base + a small dense head."""

import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.applications import vgg16
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import (
    BATCH_SIZE,
    FEATURE_EXTRACTION_MODEL_PATH,
    FIGURES_DIR,
    INPUT_SHAPE,
    SEED,
    TRAIN_DIR,
    TRAINED_MODELS_DIR,
    VAL_SPLIT,
)
from src.visualization import plot_training_curves


def build_generators():
    train_datagen = ImageDataGenerator(
        validation_split=VAL_SPLIT,
        rescale=1.0 / 255.0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    val_datagen = ImageDataGenerator(validation_split=VAL_SPLIT, rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        seed=SEED,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        target_size=INPUT_SHAPE[:2],
        subset="training",
    )
    val_generator = val_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        seed=SEED,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        target_size=INPUT_SHAPE[:2],
        subset="validation",
    )
    return train_generator, val_generator


def build_model() -> Sequential:
    vgg = vgg16.VGG16(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    output = tf.keras.layers.Flatten()(vgg.layers[-1].output)
    basemodel = Model(vgg.input, output)

    for layer in basemodel.layers:
        layer.trainable = False

    model = Sequential()
    model.add(basemodel)
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))
    return model


def exp_decay(epoch: int) -> float:
    initial_lrate = 1e-4
    k = 0.1
    return initial_lrate * np.exp(-k * epoch)


def main() -> None:
    train_generator, val_generator = build_generators()
    print(f"train_generator: {train_generator.samples} images, class_indices={train_generator.class_indices}")
    print(f"val_generator:   {val_generator.samples} images")

    model = build_model()
    model.summary()

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.RMSprop(learning_rate=1e-4),
        metrics=["accuracy"],
    )

    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        LearningRateScheduler(exp_decay),
        EarlyStopping(monitor="val_loss", patience=4, mode="min", min_delta=0.01),
        ModelCheckpoint(str(FEATURE_EXTRACTION_MODEL_PATH), monitor="val_loss", save_best_only=True, mode="min"),
    ]

    validation_steps = math.ceil(val_generator.samples / BATCH_SIZE)
    history = model.fit(
        train_generator,
        epochs=30,
        callbacks=callbacks,
        validation_data=val_generator,
        validation_steps=validation_steps,
        verbose=1,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_curves(
        history.history,
        title_prefix="Feature-Extraction Model",
        save_path_prefix=FIGURES_DIR / "feature_extraction_model",
    )
    print(f"Saved best model to {FEATURE_EXTRACTION_MODEL_PATH}")


if __name__ == "__main__":
    main()

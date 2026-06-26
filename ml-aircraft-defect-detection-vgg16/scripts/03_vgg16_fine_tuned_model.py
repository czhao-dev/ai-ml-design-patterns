#!/usr/bin/env python3
"""Fine-tune a VGG16 model by unfreezing the convolutional base from block5_conv3 onward."""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.applications import vgg16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import (
    BATCH_SIZE,
    FIGURES_DIR,
    FINE_TUNED_MODEL_PATH,
    INPUT_SHAPE,
    SEED,
    TRAIN_DIR,
    TRAINED_MODELS_DIR,
    VALID_DIR,
)
from src.visualization import plot_training_curves


def build_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        seed=SEED,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        target_size=INPUT_SHAPE[:2],
    )
    valid_generator = valid_datagen.flow_from_directory(
        directory=VALID_DIR,
        seed=SEED,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        target_size=INPUT_SHAPE[:2],
    )
    return train_generator, valid_generator


def build_model() -> Sequential:
    vgg = vgg16.VGG16(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    output = tf.keras.layers.Flatten()(vgg.layers[-1].output)
    basemodel = Model(vgg.input, output)

    set_trainable = False
    for layer in basemodel.layers:
        if layer.name == "block5_conv3":
            set_trainable = True
        layer.trainable = set_trainable

    model = Sequential()
    model.add(basemodel)
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))
    return model


def main() -> None:
    train_generator, valid_generator = build_generators()
    print(f"train_generator: {train_generator.samples} images, class_indices={train_generator.class_indices}")
    print(f"valid_generator: {valid_generator.samples} images")

    model = build_model()
    for layer in model.layers[0].layers:
        print(f"{layer.name}: trainable={layer.trainable}")
    model.summary()

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy"],
    )

    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, mode="min", min_delta=0.01, restore_best_weights=True),
        ModelCheckpoint(str(FINE_TUNED_MODEL_PATH), monitor="val_loss", save_best_only=True, mode="min"),
    ]

    history = model.fit(
        train_generator,
        epochs=30,
        callbacks=callbacks,
        validation_data=valid_generator,
        verbose=1,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_curves(
        history.history,
        title_prefix="Fine-Tuned Model",
        save_path_prefix=FIGURES_DIR / "fine_tuned_model",
    )
    print(f"Saved best model to {FINE_TUNED_MODEL_PATH}")


if __name__ == "__main__":
    main()

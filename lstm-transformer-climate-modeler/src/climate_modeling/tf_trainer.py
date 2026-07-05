"""Training loop, evaluation, and weight serialisation for TF models.

All gradient computation uses ``tf.GradientTape``; no ``tf.keras``
optimisers, losses, or callbacks are used.

Public API
----------
train_model      Mini-batch SGD loop with early stopping.
evaluate_model   Per-target metrics in original (un-normalised) units.
save_weights     Persist trainable weights to a JSON file.
load_weights     Restore trainable weights from a JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from climate_modeling.metrics import regression_metrics
from climate_modeling.sequence_dataset import SEQ_TARGET_NAMES, SequenceScaler
from climate_modeling.tf_cells import AdamOptimizer


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _mse(pred: tf.Tensor, true: tf.Tensor) -> tf.Tensor:
    """Mean squared error averaged over all elements."""
    return tf.reduce_mean(tf.square(pred - true))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_model(
    model: tf.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int = 150,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 15,
    val_fraction: float = 0.1,
    verbose: bool = True,
) -> tuple[dict[str, list[float]], dict[str, list]]:
    """Train *model* with mini-batch Adam and early stopping.

    The most-recent ``val_fraction`` of the training windows are held out
    as a **temporal** validation set (no shuffling across the split point).

    Args:
        model:         :class:`LSTMForecaster` or :class:`TransformerForecaster`.
        X_train:       ``(n, lookback, N_FEATURES)`` – already normalised.
        y_train:       ``(n, horizon, N_TARGETS)``   – already normalised.
        epochs:        Maximum number of training epochs.
        batch_size:    Mini-batch size.
        lr:            Initial Adam learning rate.
        patience:      Early-stopping patience (epochs without val improvement).
        val_fraction:  Fraction of training windows used for validation.
        verbose:       Print progress every 10 epochs.

    Returns:
        history:      ``{"train_loss": [...], "val_loss": [...]}``
        best_weights: ``{variable_name: list}`` – weights at best val epoch
                      (already restored to *model* before returning).
    """
    n = len(X_train)
    n_val = max(1, int(n * val_fraction))
    # Temporal split – last n_val windows are most recent (closest to test)
    X_tr, X_val = X_train[: n - n_val], X_train[n - n_val :]
    y_tr, y_val = y_train[: n - n_val], y_train[n - n_val :]

    optimizer = AdamOptimizer(lr=lr, name="adam")
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_weights: dict[str, list] = {}
    patience_counter = 0

    n_tr = len(X_tr)
    X_val_tf = tf.constant(X_val, dtype=tf.float32)
    y_val_tf = tf.constant(y_val, dtype=tf.float32)

    for epoch in range(epochs):
        # Shuffle training indices each epoch
        indices = np.random.permutation(n_tr)
        epoch_losses: list[float] = []

        for start in range(0, n_tr, batch_size):
            batch_idx = indices[start : start + batch_size]
            X_b = tf.constant(X_tr[batch_idx], dtype=tf.float32)
            y_b = tf.constant(y_tr[batch_idx], dtype=tf.float32)

            with tf.GradientTape() as tape:
                pred = model(X_b, training=True)
                # TransformerForecaster returns (predictions, attn_weights)
                if isinstance(pred, tuple):
                    pred = pred[0]
                loss = _mse(pred, y_b)

            grads = tape.gradient(loss, model.trainable_variables)
            # Gradient clipping to prevent exploding gradients
            clipped, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
            optimizer.apply_gradients(list(zip(clipped, model.trainable_variables)))
            epoch_losses.append(float(loss))

        train_loss = float(np.mean(epoch_losses))

        # Validation loss (no dropout)
        val_pred = model(X_val_tf, training=False)
        if isinstance(val_pred, tuple):
            val_pred = val_pred[0]
        val_loss = float(_mse(val_pred, y_val_tf))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Early stopping
        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_weights = {
                v.name: v.numpy().tolist() for v in model.trainable_variables
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch + 1}  (best val={best_val_loss:.4f})")
            break

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch + 1:>4}/{epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}"
            )

    # Restore best weights
    if best_weights:
        var_dict = {v.name: v for v in model.trainable_variables}
        for name, val_arr in best_weights.items():
            if name in var_dict:
                var_dict[name].assign(tf.constant(val_arr, dtype=tf.float32))

    return history, best_weights


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model: tf.Module,
    X: np.ndarray,
    y_norm: np.ndarray,
    scaler: SequenceScaler,
) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    """Evaluate *model* and return per-target metrics in original units.

    Predictions for PRCP and SNOW are clamped to ``[0, ∞)`` after
    inverse-transforming (matching the non-negative constraint in the
    Ridge baseline).

    Args:
        model:   Trained :class:`LSTMForecaster` or :class:`TransformerForecaster`.
        X:       ``(n_windows, lookback, N_FEATURES)`` – normalised.
        y_norm:  ``(n_windows, horizon, N_TARGETS)``   – normalised ground truth.
        scaler:  :class:`SequenceScaler` used during training.

    Returns:
        metrics: ``{target_name: {"mae": …, "rmse": …, "r2": …}}``
        pred:    ``(n_windows, horizon, N_TARGETS)`` – predictions in original units.
    """
    X_tf = tf.constant(X, dtype=tf.float32)
    raw_pred = model(X_tf, training=False)
    if isinstance(raw_pred, tuple):
        raw_pred = raw_pred[0]
    pred_norm = raw_pred.numpy()                        # (n, horizon, N_TARGETS)

    # Inverse-transform to original scale
    pred = scaler.inverse_transform_targets(pred_norm)  # (n, horizon, N_TARGETS)
    true = scaler.inverse_transform_targets(y_norm)     # (n, horizon, N_TARGETS)

    # Clamp PRCP (idx 0) and SNOW (idx 1) to non-negative
    pred[:, :, 0] = np.maximum(0.0, pred[:, :, 0])
    pred[:, :, 1] = np.maximum(0.0, pred[:, :, 1])

    results: dict[str, dict[str, float]] = {}
    for i, target in enumerate(SEQ_TARGET_NAMES):
        pred_flat = pred[:, :, i].flatten().tolist()
        true_flat = true[:, :, i].flatten().tolist()
        results[target] = regression_metrics(true_flat, pred_flat)

    return results, pred


# ---------------------------------------------------------------------------
# Weight serialisation
# ---------------------------------------------------------------------------


def save_weights(model: tf.Module, path: str | Path) -> None:
    """Write all trainable weights to *path* as a JSON file.

    Values are stored as nested Python lists (JSON-compatible).
    The file can be re-read with :func:`load_weights`.
    """
    weights = {v.name: v.numpy().tolist() for v in model.trainable_variables}
    Path(path).write_text(json.dumps(weights, indent=2), encoding="utf-8")


def load_weights(model: tf.Module, path: str | Path) -> None:
    """Restore trainable weights from a JSON file produced by :func:`save_weights`.

    Variables present in the file but absent from *model* are silently
    skipped so that partial checkpoints do not raise errors.
    """
    weights = json.loads(Path(path).read_text(encoding="utf-8"))
    var_dict = {v.name: v for v in model.trainable_variables}
    for name, val in weights.items():
        if name in var_dict:
            var_dict[name].assign(tf.constant(val, dtype=tf.float32))

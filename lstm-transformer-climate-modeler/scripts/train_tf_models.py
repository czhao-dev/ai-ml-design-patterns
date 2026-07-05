"""Train LSTM and Transformer forecasters on NOAA daily weather data.

Usage
-----
    python scripts/train_tf_models.py [OPTIONS]

Options (all have defaults)
----------------------------
    --data          Path to NOAA CSV export            [962598.csv]
    --station       Station name to model              [READING MA US]
    --train-start   Training window start date         [2012-01-01]
    --train-end     Training window end date           [2016-12-31]
    --test-start    Test window start date             [2017-01-01]
    --test-end      Test window end date               [2017-12-31]
    --lookback      Sequence input length (days)       [60]
    --horizon       Forecast horizon (days)            [7]
    --epochs        Maximum training epochs            [150]
    --batch-size    Mini-batch size                    [32]
    --lr            Adam learning rate                 [1e-3]
    --patience      Early-stopping patience            [15]
    --reports-dir   Output directory for metrics/weights/figures [reports]
    --seed          Random seed for reproducibility   [42]
    --no-lstm       Skip LSTM training
    --no-transformer Skip Transformer training
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date
from pathlib import Path

# Suppress TensorFlow C++ info / warning messages
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---------------------------------------------------------------------------
# Ensure the project package is importable when the script is run directly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np
import tensorflow as tf

from climate_modeling.data import load_station_records, parse_iso_date
from climate_modeling.sequence_dataset import (
    N_FEATURES,
    SEQ_TARGET_NAMES,
    build_sequence_dataset,
)
from climate_modeling.tf_models import LSTMForecaster, TransformerForecaster
from climate_modeling.tf_trainer import evaluate_model, save_weights, train_model


# ---------------------------------------------------------------------------
# SVG loss-curve helper (no matplotlib dependency)
# ---------------------------------------------------------------------------


def _write_loss_curve_svg(
    path: Path,
    train_losses: list[float],
    val_losses: list[float],
    title: str,
) -> None:
    """Write a minimal SVG line chart of training and validation loss."""
    w, h, pad = 640, 320, 50
    n = len(train_losses)
    all_vals = train_losses + val_losses
    y_min, y_max = min(all_vals), max(all_vals)
    y_range = max(y_max - y_min, 1e-9)

    def sx(i: int) -> float:
        return pad + (i / max(n - 1, 1)) * (w - 2 * pad)

    def sy(v: float) -> float:
        return h - pad - ((v - y_min) / y_range) * (h - 2 * pad)

    def polyline(vals: list[float], colour: str) -> str:
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals))
        return f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2"/>'

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        f'<rect width="{w}" height="{h}" fill="#fff"/>',
        f'<text x="{w//2}" y="20" text-anchor="middle" font-size="13" font-family="sans-serif">{title}</text>',
        polyline(train_losses, "#2563eb"),
        polyline(val_losses, "#dc2626"),
        f'<text x="{pad}" y="{h - 8}" font-size="10" fill="#2563eb" font-family="sans-serif">— train</text>',
        f'<text x="{pad + 60}" y="{h - 8}" font-size="10" fill="#dc2626" font-family="sans-serif">— val</text>',
        "</svg>",
    ]
    path.write_text("\n".join(svg_lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    reports_dir = Path(args.reports_dir)
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load raw records ------------------------------------------------
    print(f"Loading records from {args.data!r} …")
    records = load_station_records(args.data, args.station)
    print(f"  {len(records)} records loaded for station '{args.station}'")

    train_start = parse_iso_date(args.train_start)
    train_end = parse_iso_date(args.train_end)
    test_start = parse_iso_date(args.test_start)
    test_end = parse_iso_date(args.test_end)

    # ---- Build sequence dataset -----------------------------------------
    print(
        f"\nBuilding sequence dataset  "
        f"(lookback={args.lookback}, horizon={args.horizon}) …"
    )
    X_train, y_train, X_test, y_test, scaler = build_sequence_dataset(
        records,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        lookback=args.lookback,
        horizon=args.horizon,
    )
    print(f"  train windows : {len(X_train):>5}")
    print(f"  test  windows : {len(X_test):>5}")
    print(f"  input shape   : {X_train.shape}")
    print(f"  target shape  : {y_train.shape}")

    model_results: dict[str, dict] = {}

    # ---- LSTM -----------------------------------------------------------
    if not args.no_lstm:
        print("\n" + "=" * 60)
        print("Training LSTM forecaster")
        print("=" * 60)
        lstm = LSTMForecaster(
            input_size=N_FEATURES,
            hidden_size=64,
            n_layers=2,
            horizon=args.horizon,
            dropout_rate=0.3,
            name="lstm_forecaster",
        )
        print(f"  Trainable parameters: {_count_params(lstm):,}")

        history_lstm, _ = train_model(
            lstm,
            X_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            verbose=True,
        )

        metrics_lstm, _ = evaluate_model(lstm, X_test, y_test, scaler)
        model_results["lstm"] = {
            "architecture": "LSTMForecaster",
            "hyperparams": {
                "hidden_size": 64,
                "n_layers": 2,
                "horizon": args.horizon,
                "dropout_rate": 0.3,
            },
            "metrics": metrics_lstm,
            "epochs_trained": len(history_lstm["train_loss"]),
        }

        # Save weights and loss curve
        save_weights(lstm, reports_dir / "weights_lstm.json")
        _write_loss_curve_svg(
            figures_dir / "lstm_loss_curve.svg",
            history_lstm["train_loss"],
            history_lstm["val_loss"],
            title="LSTM – training and validation loss",
        )
        _print_metrics("LSTM", metrics_lstm, args.horizon)

    # ---- Transformer ----------------------------------------------------
    if not args.no_transformer:
        print("\n" + "=" * 60)
        print("Training Transformer forecaster")
        print("=" * 60)
        transformer = TransformerForecaster(
            input_size=N_FEATURES,
            d_model=32,
            n_heads=2,
            n_layers=3,
            d_ff=64,
            horizon=args.horizon,
            dropout_rate=0.3,
            name="transformer_forecaster",
        )
        print(f"  Trainable parameters: {_count_params(transformer):,}")

        history_tf, _ = train_model(
            transformer,
            X_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            verbose=True,
        )

        metrics_tf, _ = evaluate_model(transformer, X_test, y_test, scaler)
        model_results["transformer"] = {
            "architecture": "TransformerForecaster",
            "hyperparams": {
                "d_model": 32,
                "n_heads": 2,
                "n_layers": 3,
                "d_ff": 64,
                "horizon": args.horizon,
                "dropout_rate": 0.3,
            },
            "metrics": metrics_tf,
            "epochs_trained": len(history_tf["train_loss"]),
        }

        save_weights(transformer, reports_dir / "weights_transformer.json")
        _write_loss_curve_svg(
            figures_dir / "transformer_loss_curve.svg",
            history_tf["train_loss"],
            history_tf["val_loss"],
            title="Transformer – training and validation loss",
        )
        _print_metrics("Transformer", metrics_tf, args.horizon)

    # ---- Persist results -------------------------------------------------
    scaler_path = reports_dir / "sequence_scaler.json"
    scaler_path.write_text(json.dumps(scaler.to_dict(), indent=2), encoding="utf-8")

    tf_report = {
        "project": "ML Climate Modeling – deep learning",
        "station": args.station,
        "data_path": str(args.data),
        "train_window": [args.train_start, args.train_end],
        "test_window": [args.test_start, args.test_end],
        "train_windows": int(len(X_train)),
        "test_windows": int(len(X_test)),
        "lookback": args.lookback,
        "horizon": args.horizon,
        "targets": list(SEQ_TARGET_NAMES),
        "models": model_results,
    }
    metrics_path = reports_dir / "metrics_tf.json"
    metrics_path.write_text(json.dumps(tf_report, indent=2), encoding="utf-8")

    print(f"\nSaved TF metrics → {metrics_path}")
    print(f"Saved scaler    → {scaler_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_params(model: tf.Module) -> int:
    return sum(
        int(tf.size(v)) for v in model.trainable_variables
    )


def _print_metrics(
    model_name: str,
    metrics: dict[str, dict[str, float]],
    horizon: int,
) -> None:
    print(f"\n{model_name} — test-set metrics (all {horizon}-day forecast steps combined)")
    print(f"  {'Target':<6}  {'RMSE':>8}  {'MAE':>8}  {'R²':>8}")
    print(f"  {'------':<6}  {'----':>8}  {'---':>8}  {'--':>8}")
    for target, m in metrics.items():
        print(
            f"  {target:<6}  {m['rmse']:>8.3f}  {m['mae']:>8.3f}  {m['r2']:>8.3f}"
        )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LSTM and Transformer weather forecasters."
    )
    parser.add_argument("--data", default="4344212.csv")
    parser.add_argument("--station", default="READING, MA US")
    parser.add_argument("--train-start", default="1960-01-01")
    parser.add_argument("--train-end", default="2017-12-31")
    parser.add_argument("--test-start", default="2018-01-01")
    parser.add_argument("--test-end", default="2019-12-31")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-lstm", action="store_true")
    parser.add_argument("--no-transformer", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

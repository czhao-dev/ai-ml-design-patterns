#!/usr/bin/env python3
"""Train the naive response model (predicts P(convert), ignores treatment
entirely) and score it on the held-out test set -- this is the anti-pattern
policy the uplift models in scripts 04-06 are compared against.

Usage:
    python scripts/03_baseline_response_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, predictions
from src.baseline_model import predict_response_score, train_response_model
from src.features import align_feature_columns, load_splits


def main() -> None:
    train_df, val_df, test_df = load_splits()
    feature_cols = align_feature_columns(train_df, val_df, test_df)

    model = train_response_model(train_df, feature_cols)
    test_scores = predict_response_score(model, test_df, feature_cols)
    test_auc = roc_auc_score(test_df["conversion"], test_scores)
    print(f"Baseline response model test AUC (predicting conversion, ignoring treatment): {test_auc:.4f}")
    print(
        "Note: AUC measures how well this ranks likely converters, not who the "
        "campaign actually influences -- see scripts/07_evaluate_uplift.py for that."
    )

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODELS_DIR / "baseline_response_model.joblib")

    preds = predictions.load_or_init(test_df)
    preds["baseline_response_score"] = test_scores
    predictions.save(preds)
    print(f"Saved test-set predictions to {config.PREDICTIONS_TEST_PATH}")


if __name__ == "__main__":
    main()

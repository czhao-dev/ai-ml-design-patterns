#!/usr/bin/env python3
"""Train the causal forest uplift model (via econml's CausalForestDML) and
score it on the held-out test set.

Usage:
    python scripts/06_causal_forest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, predictions
from src.features import align_feature_columns, load_splits
from src.uplift.causal_forest import CausalForestUplift


def main() -> None:
    train_df, val_df, test_df = load_splits()
    feature_cols = align_feature_columns(train_df, val_df, test_df)

    model = CausalForestUplift(n_estimators=config.CAUSAL_FOREST_N_ESTIMATORS, seed=config.RANDOM_SEED)
    model.fit(
        train_df[feature_cols],
        train_df[config.TREATMENT_COLUMN].to_numpy(),
        train_df[config.OUTCOME_COLUMN].to_numpy(),
    )

    test_scores = model.predict_uplift(test_df[feature_cols])
    print(f"Causal forest predicted uplift on test set: mean=${test_scores.mean():.3f}, std=${test_scores.std():.3f}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODELS_DIR / "causal_forest.joblib")

    preds = predictions.load_or_init(test_df)
    preds["causal_forest_score"] = test_scores
    predictions.save(preds)
    print(f"Saved test-set predictions to {config.PREDICTIONS_TEST_PATH}")


if __name__ == "__main__":
    main()

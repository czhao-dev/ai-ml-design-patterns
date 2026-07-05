"""Shared accumulator for test-set uplift-model predictions.

Each script (03 baseline, 04 T-learner, 05 X-learner, 06 causal forest) adds
one score column to the same `predictions_test.parquet` file, so
scripts/07_evaluate_uplift.py and scripts/08_revenue_simulation.py can compare
every model side by side without retraining anything.
"""

from __future__ import annotations

import pandas as pd

from . import config


def load_or_init(test_df: pd.DataFrame) -> pd.DataFrame:
    if config.PREDICTIONS_TEST_PATH.exists():
        return pd.read_parquet(config.PREDICTIONS_TEST_PATH)
    return test_df[[config.TREATMENT_COLUMN, "visit", "conversion", "spend"]].copy()


def save(predictions: pd.DataFrame) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(config.PREDICTIONS_TEST_PATH, index=False)

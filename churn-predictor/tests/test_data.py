"""Randomization-balance checks (standardized mean difference table)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import covariate_balance_table, randomization_is_healthy, standardized_mean_difference


def _make_balanced_df(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    treatment = rng.binomial(1, 0.5, n)
    # Every covariate drawn independently of treatment -> should be balanced.
    df = pd.DataFrame(
        {
            config.TREATMENT_COLUMN: treatment,
            "recency": rng.integers(1, 12, n),
            "history": rng.uniform(0, 500, n),
            "mens": rng.binomial(1, 0.5, n),
            "womens": rng.binomial(1, 0.5, n),
            "newbie": rng.binomial(1, 0.5, n),
            "history_segment": rng.choice(["a", "b", "c"], n),
            "zip_code": rng.choice(["Urban", "Suburban", "Rural"], n),
            "channel": rng.choice(["Phone", "Web", "Multichannel"], n),
        }
    )
    return df


def test_standardized_mean_difference_near_zero_when_balanced():
    df = _make_balanced_df()
    smd = standardized_mean_difference(df, "history")
    assert smd < 0.1


def test_standardized_mean_difference_large_when_imbalanced():
    df = _make_balanced_df()
    # Force a covariate to differ sharply by treatment arm.
    df["history"] = np.where(df[config.TREATMENT_COLUMN] == 1, df["history"] + 1000, df["history"])
    smd = standardized_mean_difference(df, "history")
    assert smd > 1.0


def test_covariate_balance_table_and_health_check():
    df = _make_balanced_df()
    table = covariate_balance_table(df)
    assert set(["feature", "treated_mean", "control_mean", "smd"]).issubset(table.columns)
    assert randomization_is_healthy(table) is True

    df.loc[df[config.TREATMENT_COLUMN] == 1, "recency"] += 20
    imbalanced_table = covariate_balance_table(df)
    assert randomization_is_healthy(imbalanced_table) is False

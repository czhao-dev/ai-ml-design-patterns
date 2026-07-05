"""Feature encoding and split helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.features import (
    align_feature_columns,
    encode_features,
    get_feature_columns,
    split_train_val_test,
)


def _make_raw_df(n=500, seed=0):
    rng = np.random.default_rng(seed)
    treatment = rng.binomial(1, 0.5, n)
    return pd.DataFrame(
        {
            "recency": rng.integers(1, 12, n),
            "history_segment": rng.choice(["1) $0 - $100", "2) $100 - $200"], n),
            "history": rng.uniform(0, 500, n),
            "mens": rng.binomial(1, 0.5, n),
            "womens": rng.binomial(1, 0.5, n),
            "zip_code": rng.choice(["Urban", "Suburban", "Rural"], n),
            "newbie": rng.binomial(1, 0.5, n),
            "channel": rng.choice(["Phone", "Web", "Multichannel"], n),
            config.SEGMENT_COLUMN: np.where(treatment == 1, "Womens E-Mail", "No E-Mail"),
            config.TREATMENT_COLUMN: treatment,
            "visit": rng.binomial(1, 0.3, n),
            "conversion": rng.binomial(1, 0.1, n),
            "spend": rng.uniform(0, 50, n) * rng.binomial(1, 0.1, n),
        }
    )


def test_encode_features_one_hot_encodes_categoricals_only():
    df = _make_raw_df()
    encoded = encode_features(df)
    for cat_col in config.CATEGORICAL_FEATURES:
        assert cat_col not in encoded.columns
    # binary/numeric untouched
    for col in config.NUMERIC_FEATURES + config.BINARY_FEATURES:
        assert col in encoded.columns
    # outcomes/treatment preserved
    for col in ["visit", "conversion", "spend", config.TREATMENT_COLUMN]:
        assert col in encoded.columns


def test_get_feature_columns_excludes_outcomes_and_treatment():
    df = encode_features(_make_raw_df())
    feature_cols = get_feature_columns(df)
    for excluded in ["visit", "conversion", "spend", config.TREATMENT_COLUMN, config.SEGMENT_COLUMN]:
        assert excluded not in feature_cols


def test_split_preserves_treatment_ratio():
    df = encode_features(_make_raw_df(n=2000))
    train_df, val_df, test_df = split_train_val_test(df, val_fraction=0.15, test_fraction=0.15, seed=0)

    overall_rate = df[config.TREATMENT_COLUMN].mean()
    for split in (train_df, val_df, test_df):
        assert abs(split[config.TREATMENT_COLUMN].mean() - overall_rate) < 0.05

    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(df)
    assert abs(len(test_df) / total - 0.15) < 0.02
    assert abs(len(val_df) / total - 0.15) < 0.02


def test_align_feature_columns_returns_common_columns():
    df_a = pd.DataFrame({"f1": [1], "f2": [2], "spend": [0]})
    df_b = pd.DataFrame({"f1": [1], "f3": [3], "spend": [0]})
    common = align_feature_columns(df_a, df_b)
    assert common == ["f1"]

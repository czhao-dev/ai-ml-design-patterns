"""Feature encoding and train/val/test splitting for the Hillstrom dataset."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

# Columns that are never model features -- they're either identifiers,
# outcomes, or the raw RCT arm label (superseded by the binary `treatment`
# column derived in data.load_raw).
NON_FEATURE_COLUMNS = {
    config.SEGMENT_COLUMN,
    config.TREATMENT_COLUMN,
    "visit",
    "conversion",
    "spend",
}


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical covariates; leave binary/numeric covariates
    as-is. Returns a new dataframe with the original outcome/treatment/segment
    columns preserved alongside the encoded features."""
    encoded = pd.get_dummies(df, columns=config.CATEGORICAL_FEATURES, prefix=config.CATEGORICAL_FEATURES)
    return encoded


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Every column in an encoded dataframe that is a model feature (i.e. not
    an outcome, the treatment indicator, or the raw segment label)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def split_train_val_test(
    df: pd.DataFrame,
    val_fraction: float = config.VAL_FRACTION,
    test_fraction: float = config.TEST_FRACTION,
    seed: int = config.RANDOM_SEED,
):
    """Stratify by treatment so every split preserves the RCT's treated/control
    ratio -- important since the downstream uplift metrics rely on comparing
    treated vs. control outcomes within each split."""
    train_frac_of_remainder = 1.0 - (val_fraction / (1.0 - test_fraction))
    train_val_df, test_df = train_test_split(
        df, test_size=test_fraction, stratify=df[config.TREATMENT_COLUMN], random_state=seed
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=(1.0 - train_frac_of_remainder),
        stratify=train_val_df[config.TREATMENT_COLUMN],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_and_split(raw_df: pd.DataFrame):
    """Full feature-engineering pipeline: encode, then split."""
    encoded = encode_features(raw_df)
    return split_train_val_test(encoded)


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(config.TRAIN_PATH, index=False)
    val_df.to_parquet(config.VAL_PATH, index=False)
    test_df.to_parquet(config.TEST_PATH, index=False)


def load_splits():
    train_df = pd.read_parquet(config.TRAIN_PATH)
    val_df = pd.read_parquet(config.VAL_PATH)
    test_df = pd.read_parquet(config.TEST_PATH)
    return train_df, val_df, test_df


def align_feature_columns(*dfs: pd.DataFrame) -> list[str]:
    """Feature columns common to every dataframe passed in, in a fixed order
    -- guards against one-hot splits producing slightly different dummy
    columns across train/val/test (e.g. a rare zip_code level missing from
    one split)."""
    common = set(get_feature_columns(dfs[0]))
    for other in dfs[1:]:
        common &= set(get_feature_columns(other))
    return sorted(common)

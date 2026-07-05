#!/usr/bin/env python3
"""Encode features and create the train/val/test splits.

Usage:
    python scripts/02_build_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import load_raw
from src.features import build_and_split, get_feature_columns, save_splits


def main() -> None:
    df = load_raw()
    train_df, val_df, test_df = build_and_split(df)
    save_splits(train_df, val_df, test_df)

    feature_cols = get_feature_columns(train_df)
    print(f"Train: {len(train_df):,} rows | Val: {len(val_df):,} rows | Test: {len(test_df):,} rows")
    print(f"{len(feature_cols)} feature columns: {feature_cols}")
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rate = split[config.TREATMENT_COLUMN].mean()
        print(f"  {name}: treatment rate = {rate:.3f}")
    print(f"Saved splits to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()

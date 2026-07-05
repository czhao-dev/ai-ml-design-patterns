#!/usr/bin/env python3
"""Download the Hillstrom RCT dataset and verify randomization is intact.

Usage:
    python scripts/01_download_and_validate_rct.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import covariate_balance_table, download_hillstrom, load_raw, randomization_is_healthy


def main() -> None:
    path = download_hillstrom()
    print(f"Raw data at {path}")

    df = load_raw(path)
    print(f"Loaded {len(df):,} rows.")
    print(df[config.SEGMENT_COLUMN].value_counts())
    print(f"Treatment (any email) rate: {df[config.TREATMENT_COLUMN].mean():.3f}")

    balance = covariate_balance_table(df)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    balance_path = config.REPORTS_DIR / "covariate_balance.csv"
    balance.to_csv(balance_path, index=False)

    print("\nCovariate balance (treated vs. control), sorted by |SMD| descending:")
    print(balance.to_string(index=False))

    healthy = randomization_is_healthy(balance)
    print(f"\nRandomization healthy (all |SMD| < 0.1): {healthy}")
    if not healthy:
        print(
            "WARNING: some covariates are imbalanced across arms -- treat "
            "downstream causal estimates with caution."
        )
    print(f"Saved balance table to {balance_path}")


if __name__ == "__main__":
    main()

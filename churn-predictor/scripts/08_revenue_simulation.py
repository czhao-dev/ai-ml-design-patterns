#!/usr/bin/env python3
"""Translate each model's targeting policy into a $ net-incremental-revenue
number, compared against blanket (target everyone) and random targeting.

Usage:
    python scripts/08_revenue_simulation.py [--cost-per-offer 2.0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.evaluation.revenue import evaluate_random_policy, evaluate_targeting_policy

MODEL_SCORE_COLUMNS = {
    "baseline_response_score": "Baseline response model (ignores treatment)",
    "t_learner_score": "T-learner",
    "x_learner_score": "X-learner",
    "causal_forest_score": "Causal forest",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="$ net-incremental-revenue policy comparison.")
    parser.add_argument("--cost-per-offer", type=float, default=config.COST_PER_OFFER_USD)
    args = parser.parse_args()

    preds = pd.read_parquet(config.PREDICTIONS_TEST_PATH)

    all_rows = []
    for col, label in MODEL_SCORE_COLUMNS.items():
        if col not in preds.columns:
            print(f"Skipping '{label}' -- column '{col}' not found (run its training script first).")
            continue
        policy_df = evaluate_targeting_policy(preds, col, cost_per_offer=args.cost_per_offer)
        policy_df.insert(0, "policy", label)
        all_rows.append(policy_df)

    random_df = evaluate_random_policy(preds, cost_per_offer=args.cost_per_offer)
    random_df.insert(0, "policy", "Random targeting")
    all_rows.append(random_df)

    comparison = pd.concat(all_rows, ignore_index=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.REPORTS_DIR / "revenue_policy_comparison.csv"
    comparison.to_csv(out_path, index=False)

    print(f"Cost per offer: ${args.cost_per_offer:.2f}\n")
    at_20pct = comparison[comparison["targeting_fraction"] == 0.2].sort_values("net_revenue", ascending=False)
    print("Net revenue at 20% targeting fraction, by policy (blanket = 100% row in the full CSV):")
    print(
        at_20pct[["policy", "n_targeted", "total_incremental_revenue", "total_cost", "net_revenue"]].to_string(
            index=False
        )
    )
    print(f"\nFull comparison across all targeting fractions saved to {out_path}")


if __name__ == "__main__":
    main()

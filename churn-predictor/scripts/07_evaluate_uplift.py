#!/usr/bin/env python3
"""Compute Qini curves, AUUC, uplift@20%, and per-decile calibration for
every scored model in predictions_test.parquet, and save a comparison figure
and table.

Usage:
    python scripts/07_evaluate_uplift.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.evaluation.metrics import auuc, calibration_by_decile, uplift_at_k
from src.evaluation.qini import qini_coefficient, qini_curve

MODEL_SCORE_COLUMNS = {
    "baseline_response_score": "Baseline response model (ignores treatment)",
    "t_learner_score": "T-learner",
    "x_learner_score": "X-learner",
    "causal_forest_score": "Causal forest",
}


def main() -> None:
    preds = pd.read_parquet(config.PREDICTIONS_TEST_PATH)
    y = preds[config.OUTCOME_COLUMN].to_numpy()
    treatment = preds[config.TREATMENT_COLUMN].to_numpy()

    # The Qini curve's final point (full population targeted) is the same
    # regardless of ranking, so compute it once for the random-targeting
    # reference line.
    n = len(preds)
    fractions_ref = np.arange(0, n + 1) / n
    _, ref_gains = qini_curve(y, treatment, treatment)
    total_gain = ref_gains[-1]

    summary_rows = []
    plt.figure(figsize=(7, 5))
    for col, label in MODEL_SCORE_COLUMNS.items():
        if col not in preds.columns:
            print(f"Skipping '{label}' -- column '{col}' not found (run its training script first).")
            continue
        score = preds[col].to_numpy()
        fractions, gains = qini_curve(y, treatment, score)
        plt.plot(fractions, gains, label=label)

        summary_rows.append(
            {
                "model": label,
                "qini_coefficient": qini_coefficient(y, treatment, score),
                "auuc": auuc(y, treatment, score),
                "uplift_at_20pct": uplift_at_k(y, treatment, score, 0.2),
            }
        )

        decile_table = calibration_by_decile(y, treatment, score, n_bins=config.N_QINI_BINS)
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        decile_table.to_csv(config.REPORTS_DIR / f"calibration_{col}.csv", index=False)

    plt.plot(fractions_ref, fractions_ref * total_gain, "k--", label="Random targeting")
    plt.xlabel("Fraction of customers targeted (ranked by predicted uplift)")
    plt.ylabel("Cumulative incremental spend ($)")
    plt.title("Qini curves: predicted-uplift ranking vs. random targeting")
    plt.legend()
    plt.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = config.FIGURES_DIR / "qini_curves.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved Qini curve figure to {fig_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = config.REPORTS_DIR / "uplift_model_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"Saved model comparison table to {summary_path}")


if __name__ == "__main__":
    main()

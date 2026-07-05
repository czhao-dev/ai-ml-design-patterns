"""Uplift-specific evaluation metrics beyond the Qini curve: AUUC, uplift@k,
and per-decile calibration.

None of these substitute for the fact that individual treatment effects are
fundamentally unobservable (we only ever see one of Y(1)/Y(0) per customer).
What *is* observable and testable on held-out RCT data is the *average*
realized incremental outcome within a group -- so every metric here is built
from group-level treated-vs-control comparisons, never a per-row "error".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .qini import qini_curve

# numpy 2.0 renamed trapz -> trapezoid; keep this working on either version.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def auuc(y: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray) -> float:
    """Area under the (raw, non-baseline-subtracted) uplift/Qini curve."""
    fractions, gains = qini_curve(y, treatment, uplift_score)
    return float(_trapezoid(gains, fractions))


def uplift_at_k(y: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray, k: float) -> float:
    """Realized incremental outcome (treated mean - control mean) among the
    top k-fraction of customers ranked by predicted uplift."""
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment)
    uplift_score = np.asarray(uplift_score, dtype=float)
    n = len(y)
    n_top = max(1, int(round(k * n)))
    idx = np.argsort(-uplift_score)[:n_top]
    y_top, t_top = y[idx], treatment[idx]
    treated = y_top[t_top == 1]
    control = y_top[t_top == 0]
    if len(treated) == 0 or len(control) == 0:
        return float("nan")
    return float(treated.mean() - control.mean())


def calibration_by_decile(
    y: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Bucket customers into deciles by predicted uplift (decile 1 = highest
    predicted uplift) and compare each bucket's mean predicted uplift against
    its realized incremental outcome. A well-calibrated model should show
    realized uplift decreasing monotonically (or close to it) from decile 1
    to decile n_bins, tracking the predicted values."""
    df = pd.DataFrame(
        {"y": np.asarray(y, dtype=float), "treatment": np.asarray(treatment), "score": np.asarray(uplift_score, dtype=float)}
    )
    ranks = df["score"].rank(method="first", ascending=False)
    df["decile"] = pd.qcut(ranks, n_bins, labels=False) + 1

    rows = []
    for decile, group in df.groupby("decile"):
        treated = group.loc[group["treatment"] == 1, "y"]
        control = group.loc[group["treatment"] == 0, "y"]
        realized = treated.mean() - control.mean() if len(treated) and len(control) else float("nan")
        rows.append(
            {
                "decile": int(decile),
                "n": len(group),
                "predicted_uplift": group["score"].mean(),
                "realized_uplift": realized,
            }
        )
    return pd.DataFrame(rows).sort_values("decile").reset_index(drop=True)

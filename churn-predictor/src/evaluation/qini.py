"""Qini curve and Qini coefficient (Radcliffe & Surry, 2011).

The Qini curve orders customers by descending predicted uplift and plots,
at each cumulative fraction of the population, the incremental outcome
gained by treating that top slice -- adjusted for any treated/control
imbalance within the slice, since Hillstrom's RCT is not a 50/50 split.

At cumulative count i (top-i customers by predicted uplift):

    g(i) = sum(y | treated, top-i) - sum(y | control, top-i) * (n_treated(i) / n_control(i))

g(i) is directly interpretable as "total incremental outcome from treating
these i customers, if the treated:control response ratio observed here holds
for the whole slice". A model that ranks by true uplift will front-load the
customers who benefit most, producing a curve that rises above the diagonal
line connecting (0, 0) to (N, g(N)) -- the expected curve for random
targeting. The Qini coefficient is the area between the model's curve and
that random-targeting diagonal: positive means the model beats random
targeting, larger is better, zero means no better than random.
"""

from __future__ import annotations

import numpy as np

# numpy 2.0 renamed trapz -> trapezoid; keep this working on either version.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def qini_curve(y: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray):
    """Returns (fractions, gains): gains[i] is the cumulative incremental
    outcome from treating the top fractions[i] of the population, ranked by
    descending predicted uplift. Both arrays start at (0, 0)."""
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    uplift_score = np.asarray(uplift_score, dtype=float)
    n = len(y)

    order = np.argsort(-uplift_score)
    y_sorted = y[order]
    t_sorted = treatment[order]

    cum_n_treated = np.cumsum(t_sorted)
    cum_n_control = np.cumsum(1.0 - t_sorted)
    cum_y_treated = np.cumsum(y_sorted * t_sorted)
    cum_y_control = np.cumsum(y_sorted * (1.0 - t_sorted))

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(cum_n_control > 0, cum_n_treated / cum_n_control, 0.0)
    gains = cum_y_treated - cum_y_control * ratio

    fractions = np.arange(1, n + 1) / n
    fractions = np.concatenate(([0.0], fractions))
    gains = np.concatenate(([0.0], gains))
    return fractions, gains


def qini_coefficient(y: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray) -> float:
    """Area between the model's Qini curve and the random-targeting diagonal.
    Positive => the model's ranking beats random targeting."""
    fractions, gains = qini_curve(y, treatment, uplift_score)
    random_gains = fractions * gains[-1]
    area_model = _trapezoid(gains, fractions)
    area_random = _trapezoid(random_gains, fractions)
    return float(area_model - area_random)

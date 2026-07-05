"""uplift_at_k and calibration_by_decile."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import auuc, calibration_by_decile, uplift_at_k
from src.evaluation.qini import qini_curve

Y = np.array([10.0, 2.0, 1.0, 5.0])
TREATMENT = np.array([1, 0, 1, 0])
SCORE = np.array([3.0, 2.0, 1.0, 0.0])


def test_uplift_at_k_top_half():
    # Top 2 by score: customer 1 (treated, y=10), customer 2 (control, y=2).
    result = uplift_at_k(Y, TREATMENT, SCORE, k=0.5)
    assert result == pytest.approx(10.0 - 2.0)


def test_uplift_at_k_full_population_equals_ate():
    result = uplift_at_k(Y, TREATMENT, SCORE, k=1.0)
    ate = Y[TREATMENT == 1].mean() - Y[TREATMENT == 0].mean()
    assert result == pytest.approx(ate)


def test_auuc_matches_qini_curve_area():
    fractions, gains = qini_curve(Y, TREATMENT, SCORE)
    expected = (getattr(np, "trapezoid", None) or np.trapz)(gains, fractions)
    assert auuc(Y, TREATMENT, SCORE) == pytest.approx(expected)


def test_calibration_by_decile_shape_and_columns():
    rng = np.random.default_rng(2)
    n = 200
    treatment = rng.binomial(1, 0.5, n)
    score = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)

    table = calibration_by_decile(y, treatment, score, n_bins=5)
    assert list(table.columns) == ["decile", "n", "predicted_uplift", "realized_uplift"]
    assert list(table["decile"]) == [1, 2, 3, 4, 5]
    assert table["n"].sum() == n



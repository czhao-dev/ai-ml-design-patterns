"""Qini curve/coefficient against a hand-computed toy example.

Four customers, scores already in descending order:

| customer | treatment | y  | score |
|---|---|---|---|
| 1 | treated | 10 | 3 |
| 2 | control | 2  | 2 |
| 3 | treated | 1  | 1 |
| 4 | control | 5  | 0 |

Cumulative gain g(i) = cum_y_treated(i) - cum_y_control(i) * (n_treated(i)/n_control(i)):

i=1: 10 - 0        = 10
i=2: 10 - 2*(1/1)  = 8
i=3: 11 - 2*(2/1)  = 7
i=4: 11 - 7*(2/2)  = 4
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.qini import qini_coefficient, qini_curve

Y = np.array([10.0, 2.0, 1.0, 5.0])
TREATMENT = np.array([1, 0, 1, 0])
SCORE = np.array([3.0, 2.0, 1.0, 0.0])


def test_qini_curve_matches_hand_computed_gains():
    fractions, gains = qini_curve(Y, TREATMENT, SCORE)
    np.testing.assert_allclose(fractions, [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(gains, [0.0, 10.0, 8.0, 7.0, 4.0])


def test_qini_coefficient_matches_hand_computed_area():
    # area_model = 6.75 (trapezoidal area under the gains curve above),
    # area_random = 2.0 (trapezoidal area under the 0->4 diagonal).
    coef = qini_coefficient(Y, TREATMENT, SCORE)
    assert coef == pytest.approx(4.75, abs=1e-9)


def test_qini_coefficient_discriminating_score_beats_random_score(synthetic_rct):
    # A score that actually tracks the true treatment effect should produce a
    # much larger (more positive) Qini coefficient than a score with no
    # relationship to the outcome or treatment at all.
    X, treatment, y, true_effect = synthetic_rct
    rng = np.random.default_rng(1)
    random_score = rng.normal(0, 1, len(y))

    coef_true = qini_coefficient(y, treatment, true_effect)
    coef_random = qini_coefficient(y, treatment, random_score)

    assert coef_true > 0
    assert coef_true > abs(coef_random) * 5

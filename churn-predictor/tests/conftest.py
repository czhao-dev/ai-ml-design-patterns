"""Shared test fixtures for churn-predictor.

`synthetic_rct` is a synthetic randomized trial with a known closed-form
treatment effect, used to check that the hand-rolled T-learner/X-learner
recover something close to the true CATE function -- the same kind of check
this repo's other from-scratch implementations (arena planner, LSTM cells)
get validated against.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest


def true_cate(x1: np.ndarray) -> np.ndarray:
    """Ground-truth treatment effect used by `synthetic_rct`: linear in x1
    only (x2 affects the baseline outcome but not the treatment effect)."""
    return 2.0 * x1 - 1.0


@pytest.fixture
def synthetic_rct():
    rng = np.random.default_rng(0)
    n = 3000
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    treatment = rng.binomial(1, 0.5, n)

    baseline = 3.0 * x1 + 2.0 * x2
    effect = true_cate(x1)
    noise = rng.normal(0, 0.5, n)
    y = baseline + treatment * effect + noise

    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, treatment, y, effect

"""X-learner correctness: recovers a known synthetic CATE within tolerance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.uplift.x_learner import XLearner


def test_x_learner_recovers_known_cate(synthetic_rct):
    X, treatment, y, true_effect = synthetic_rct
    model = XLearner(seed=0)
    model.fit(X, treatment, y)

    predicted = model.predict_uplift(X)
    mae = np.abs(predicted - true_effect).mean()
    assert mae < 0.35, f"X-learner mean absolute CATE error too high: {mae:.3f}"


def test_x_learner_predict_before_fit_raises():
    model = XLearner()
    with pytest.raises(RuntimeError):
        model.predict_uplift(pd.DataFrame({"x1": [0.1], "x2": [0.2]}))

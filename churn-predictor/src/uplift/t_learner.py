"""T-learner: the simplest two-model CATE estimator.

Fit one regressor on treated customers only (mu1), one on control customers
only (mu0), each predicting the outcome from covariates. The estimated
treatment effect for a new customer is just the difference of the two models'
predictions: tau(x) = mu1(x) - mu0(x).

Cheap and easy to reason about, but each model only ever sees half the data
and the difference of two independently-fit regressions can be noisy where
the two functions are both changing quickly -- exactly the gap the X-learner
in this package is designed to close.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .base import UpliftModel


class TLearner(UpliftModel):
    def __init__(self, base_learner_factory=None, seed: int = 42):
        # functools.partial (rather than a lambda) so a fitted TLearner can be
        # joblib-pickled -- lambdas aren't picklable.
        self.base_learner_factory = base_learner_factory or partial(
            GradientBoostingRegressor, random_state=seed
        )
        self.model_treated = None
        self.model_control = None

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> "TLearner":
        treatment = np.asarray(treatment)
        y = np.asarray(y)
        mask_treated = treatment == 1

        self.model_treated = self.base_learner_factory()
        self.model_treated.fit(X.loc[mask_treated], y[mask_treated])

        self.model_control = self.base_learner_factory()
        self.model_control.fit(X.loc[~mask_treated], y[~mask_treated])
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_treated is None or self.model_control is None:
            raise RuntimeError("TLearner.fit must be called before predict_uplift.")
        return self.model_treated.predict(X) - self.model_control.predict(X)

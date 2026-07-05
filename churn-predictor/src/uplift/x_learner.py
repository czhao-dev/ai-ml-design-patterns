"""X-learner (Kunzel et al., 2019): a four-model CATE estimator that reuses
the *other* arm's outcome model to impute individual treatment effects before
regressing on them, then blends the two resulting effect models by estimated
propensity.

Stage 1 (same as the T-learner): fit mu1(x) on treated customers, mu0(x) on
control customers.

Stage 2: for each treated customer, impute their (unobservable) control
outcome as mu0(x) and take the imputed effect D1_i = Y_i - mu0(X_i). For each
control customer, impute their (unobservable) treated outcome as mu1(x) and
take D0_i = mu1(X_i) - Y_i. Regress D1 on X (giving tau1) and D0 on X (giving
tau0) -- both are now direct estimates of the treatment effect function,
each built from one arm's data but using the *other* arm's model as its
outcome proxy.

Stage 3: combine tau1 and tau0 using an estimated propensity score g(x) =
P(treatment=1 | X): tau(x) = g(x) * tau0(x) + (1 - g(x)) * tau1(x). Weighting
by propensity means arms with fewer observations (and therefore noisier
tau estimates) get down-weighted where they'd otherwise dominate -- this is
the main advantage of X- over T-learner in datasets with treatment-group
imbalance (Hillstrom's ~1:2 no-email:email split is a mild version of this).
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from .base import UpliftModel


class XLearner(UpliftModel):
    def __init__(self, base_learner_factory=None, propensity_factory=None, seed: int = 42):
        # functools.partial (rather than a lambda) so a fitted XLearner can be
        # joblib-pickled -- lambdas aren't picklable.
        self.base_learner_factory = base_learner_factory or partial(
            GradientBoostingRegressor, random_state=seed
        )
        self.propensity_factory = propensity_factory or partial(
            LogisticRegression, max_iter=1000, random_state=seed
        )
        self.mu1 = None
        self.mu0 = None
        self.tau1 = None
        self.tau0 = None
        self.propensity_model = None

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> "XLearner":
        treatment = np.asarray(treatment)
        y = np.asarray(y, dtype=float)
        mask_treated = treatment == 1

        X_treated, y_treated = X.loc[mask_treated], y[mask_treated]
        X_control, y_control = X.loc[~mask_treated], y[~mask_treated]

        # Stage 1: outcome models per arm.
        self.mu1 = self.base_learner_factory().fit(X_treated, y_treated)
        self.mu0 = self.base_learner_factory().fit(X_control, y_control)

        # Stage 2: imputed treatment effects, regressed on covariates.
        d1 = y_treated - self.mu0.predict(X_treated)
        d0 = self.mu1.predict(X_control) - y_control
        self.tau1 = self.base_learner_factory().fit(X_treated, d1)
        self.tau0 = self.base_learner_factory().fit(X_control, d0)

        # Stage 3: propensity model for blending.
        self.propensity_model = self.propensity_factory().fit(X, treatment)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        if self.tau1 is None or self.tau0 is None:
            raise RuntimeError("XLearner.fit must be called before predict_uplift.")
        g = self.propensity_model.predict_proba(X)[:, 1]
        g = np.clip(g, 0.01, 0.99)
        return g * self.tau0.predict(X) + (1.0 - g) * self.tau1.predict(X)

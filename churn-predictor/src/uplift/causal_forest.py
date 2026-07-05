"""Causal forest CATE estimator, via econml's `CausalForestDML`.

A from-scratch honest-splitting causal forest (Athey & Wager, 2019) is a
research-grade tree-ensemble with its own sample-splitting and asymptotic
variance machinery -- reimplementing it correctly is a multi-week project on
its own with a real risk of subtle, hard-to-detect bugs in exactly the part
that matters (honest splits). This project uses `econml`'s implementation
here deliberately, in contrast to the hand-rolled T-/X-learners, matching
this repo's general pattern of implementing from scratch where it's
educational and reaching for a battle-tested library where correctness risk
dominates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import UpliftModel


class CausalForestUplift(UpliftModel):
    def __init__(self, n_estimators: int = 500, seed: int = 42):
        self.n_estimators = n_estimators
        self.seed = seed
        self.model = None

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> "CausalForestUplift":
        try:
            from econml.dml import CausalForestDML
        except ImportError as exc:  # pragma: no cover - exercised only when econml is missing
            raise ImportError(
                "econml is required for CausalForestUplift. Install it with "
                "`pip install econml` (see requirements.txt)."
            ) from exc
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        self.model = CausalForestDML(
            model_y=GradientBoostingRegressor(random_state=self.seed),
            model_t=GradientBoostingClassifier(random_state=self.seed),
            discrete_treatment=True,
            n_estimators=self.n_estimators,
            random_state=self.seed,
        )
        self.model.fit(Y=np.asarray(y, dtype=float), T=np.asarray(treatment), X=X)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CausalForestUplift.fit must be called before predict_uplift.")
        return np.asarray(self.model.effect(X)).reshape(-1)

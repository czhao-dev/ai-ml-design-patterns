"""The naive baseline: a standard supervised classifier that predicts P(convert)
directly from customer features, with no awareness of the treatment at all.

This is the anti-pattern uplift modeling is meant to fix. Scoring and targeting
customers by "likelihood to convert" tends to concentrate the campaign on
people who would have converted anyway (the "sure things") -- the campaign
gets credit for outcomes it didn't cause. `scripts/07_evaluate_uplift.py` and
`scripts/08_revenue_simulation.py` both score this baseline as a policy
side-by-side with the uplift models specifically to make that failure mode
visible in the results, not just assert it.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from . import config


def train_response_model(
    train_df: pd.DataFrame, feature_cols: list[str], seed: int = config.RANDOM_SEED
) -> GradientBoostingClassifier:
    """Fit P(conversion=1 | X) on the full training population, ignoring
    treatment entirely -- exactly what a "who is likely to churn/convert"
    classifier looks like in isolation."""
    model = GradientBoostingClassifier(random_state=seed)
    model.fit(train_df[feature_cols], train_df["conversion"])
    return model


def predict_response_score(model: GradientBoostingClassifier, df: pd.DataFrame, feature_cols: list[str]):
    """P(conversion=1 | X) for each row -- used as the ranking score for the
    naive "target likely converters" policy."""
    return model.predict_proba(df[feature_cols])[:, 1]

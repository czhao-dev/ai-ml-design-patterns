"""Translate a targeting policy into a $ net-incremental-revenue number.

Because Hillstrom is a genuine RCT, the treated-vs-control difference in mean
spend within any subgroup (e.g. "the top 20% of customers by predicted
uplift") is an unbiased estimate of that subgroup's average treatment effect
-- no simulation or invented ground truth required. This module compares:

- **Uplift-targeted policies**: treat only the top k% ranked by a model's
  predicted uplift.
- **Response-model-targeted policies**: treat the top k% ranked by predicted
  P(convert), ignoring treatment -- the naive baseline's policy.
- **Random targeting**: treat a random k% (in expectation, identical to the
  population's overall average treatment effect).
- **Blanket targeting**: treat everyone (k=100%, the "no model at all" status
  quo policy many retention teams actually run).

All under an assumed per-customer offer cost, so the tradeoff between
targeting precision and campaign cost is explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


def realized_incremental_outcome(
    df: pd.DataFrame, outcome_col: str = config.OUTCOME_COLUMN, treatment_col: str = config.TREATMENT_COLUMN
) -> float:
    """Mean(outcome | treated) - Mean(outcome | control) within `df`. Valid as
    an unbiased effect estimate as long as treatment assignment within `df`
    was randomized (true here, since `df` is just a subset of an RCT)."""
    treated = df.loc[df[treatment_col] == 1, outcome_col]
    control = df.loc[df[treatment_col] == 0, outcome_col]
    if len(treated) == 0 or len(control) == 0:
        return float("nan")
    return float(treated.mean() - control.mean())


def evaluate_targeting_policy(
    df: pd.DataFrame,
    score_col: str,
    cost_per_offer: float = config.COST_PER_OFFER_USD,
    k_grid=None,
    outcome_col: str = config.OUTCOME_COLUMN,
    treatment_col: str = config.TREATMENT_COLUMN,
) -> pd.DataFrame:
    """For each targeting fraction k in `k_grid`, target the top-k customers
    ranked by `score_col` (descending) and report the realized incremental
    revenue, campaign cost, and net revenue for that policy."""
    k_grid = k_grid if k_grid is not None else config.TARGETING_FRACTIONS
    n = len(df)
    ranks = df[score_col].rank(method="first", ascending=False)

    rows = []
    for k in k_grid:
        n_targeted = max(1, int(round(k * n)))
        bucket = df.loc[ranks <= n_targeted]
        incremental_per_customer = realized_incremental_outcome(bucket, outcome_col, treatment_col)
        total_incremental_revenue = incremental_per_customer * n_targeted
        total_cost = cost_per_offer * n_targeted
        rows.append(
            {
                "targeting_fraction": k,
                "n_targeted": n_targeted,
                "incremental_revenue_per_customer": incremental_per_customer,
                "total_incremental_revenue": total_incremental_revenue,
                "total_cost": total_cost,
                "net_revenue": total_incremental_revenue - total_cost,
            }
        )
    return pd.DataFrame(rows)


def evaluate_random_policy(
    df: pd.DataFrame,
    cost_per_offer: float = config.COST_PER_OFFER_USD,
    k_grid=None,
    outcome_col: str = config.OUTCOME_COLUMN,
    treatment_col: str = config.TREATMENT_COLUMN,
) -> pd.DataFrame:
    """Random targeting baseline. In expectation a uniformly random subset of
    any size has the same per-customer incremental outcome as the whole
    population's average treatment effect, so this uses the population ATE
    directly rather than resampling."""
    k_grid = k_grid if k_grid is not None else config.TARGETING_FRACTIONS
    ate = realized_incremental_outcome(df, outcome_col, treatment_col)
    n = len(df)

    rows = []
    for k in k_grid:
        n_targeted = max(1, int(round(k * n)))
        total_incremental_revenue = ate * n_targeted
        total_cost = cost_per_offer * n_targeted
        rows.append(
            {
                "targeting_fraction": k,
                "n_targeted": n_targeted,
                "incremental_revenue_per_customer": ate,
                "total_incremental_revenue": total_incremental_revenue,
                "total_cost": total_cost,
                "net_revenue": total_incremental_revenue - total_cost,
            }
        )
    return pd.DataFrame(rows)

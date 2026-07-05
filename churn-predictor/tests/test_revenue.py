"""Revenue policy comparison against a hand-computed toy example."""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.revenue import evaluate_random_policy, evaluate_targeting_policy, realized_incremental_outcome

# 4 customers: treated spend 10 & 6, control spend 2 & 4.
DF = pd.DataFrame(
    {
        "treatment": [1, 0, 1, 0],
        "spend": [10.0, 2.0, 6.0, 4.0],
        "score": [4.0, 3.0, 2.0, 1.0],
    }
)


def test_realized_incremental_outcome_full_population():
    # treated mean = 8, control mean = 3 -> ATE = 5
    result = realized_incremental_outcome(DF, outcome_col="spend", treatment_col="treatment")
    assert result == pytest.approx(5.0)


def test_evaluate_targeting_policy_top_half():
    # k=0.5 -> top 2 by score: customer 1 (treated, 10), customer 2 (control, 2).
    policy = evaluate_targeting_policy(DF, "score", cost_per_offer=1.0, k_grid=[0.5])
    row = policy.iloc[0]
    assert row["n_targeted"] == 2
    assert row["incremental_revenue_per_customer"] == pytest.approx(10.0 - 2.0)
    assert row["total_incremental_revenue"] == pytest.approx(16.0)
    assert row["total_cost"] == pytest.approx(2.0)
    assert row["net_revenue"] == pytest.approx(14.0)


def test_evaluate_targeting_policy_full_population_matches_ate():
    policy = evaluate_targeting_policy(DF, "score", cost_per_offer=0.5, k_grid=[1.0])
    row = policy.iloc[0]
    assert row["n_targeted"] == 4
    assert row["incremental_revenue_per_customer"] == pytest.approx(5.0)
    assert row["total_incremental_revenue"] == pytest.approx(20.0)
    assert row["total_cost"] == pytest.approx(2.0)
    assert row["net_revenue"] == pytest.approx(18.0)


def test_evaluate_random_policy_uses_population_ate():
    policy = evaluate_random_policy(DF, cost_per_offer=1.0, k_grid=[0.5])
    row = policy.iloc[0]
    assert row["n_targeted"] == 2
    assert row["incremental_revenue_per_customer"] == pytest.approx(5.0)  # population ATE, not bucket-specific
    assert row["total_incremental_revenue"] == pytest.approx(10.0)
    assert row["net_revenue"] == pytest.approx(8.0)

"""Evaluation metrics for regression forecasts."""

from __future__ import annotations

import math


def regression_metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    """Return MAE, RMSE, and R-squared."""

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays must have equal length.")
    if not actual:
        raise ValueError("Cannot score empty arrays.")

    errors = [truth - forecast for truth, forecast in zip(actual, predicted)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error**2 for error in errors) / len(errors))

    actual_mean = sum(actual) / len(actual)
    total_sum_squares = sum((truth - actual_mean) ** 2 for truth in actual)
    residual_sum_squares = sum(error**2 for error in errors)
    r2 = 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares else 0.0

    return {"mae": mae, "rmse": rmse, "r2": r2}

"""RMSE/MAE/R^2 helpers, shared by both tracks' rating-regression evaluation."""

import math


def rmse(predictions, targets):
    n = len(predictions)
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(predictions, targets)) / n)


def mae(predictions, targets):
    n = len(predictions)
    return sum(abs(p - t) for p, t in zip(predictions, targets)) / n


def r_squared(predictions, targets):
    mean_target = sum(targets) / len(targets)
    ss_res = sum((t - p) ** 2 for p, t in zip(predictions, targets))
    ss_tot = sum((t - mean_target) ** 2 for t in targets)
    if ss_tot < 1e-12:
        return float("nan")
    return 1 - ss_res / ss_tot

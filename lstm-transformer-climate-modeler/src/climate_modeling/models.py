"""Small, dependency-free forecasting models."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from math import sqrt

from climate_modeling.data import WeatherRecord


@dataclass
class RidgeRegressor:
    """Linear ridge regression with train-set standardization."""

    alpha: float = 1.0
    non_negative: bool = False
    coefficients: list[float] | None = None
    means: list[float] | None = None
    scales: list[float] | None = None

    def fit(self, features: list[list[float]], target: list[float]) -> "RidgeRegressor":
        _validate_feature_matrix(features)
        if len(features) != len(target):
            raise ValueError("Feature and target row counts must match.")
        if self.alpha < 0:
            raise ValueError("Ridge alpha must be non-negative.")

        self.means, self.scales = _column_stats(features)
        design = [[1.0] + self._standardize(row) for row in features]
        xtx, xty = _normal_equations(design, target)

        for diagonal in range(1, len(xtx)):
            xtx[diagonal][diagonal] += self.alpha

        self.coefficients = _solve_linear_system(xtx, xty)
        return self

    def predict(self, features: list[list[float]]) -> list[float]:
        if self.coefficients is None:
            raise ValueError("Model has not been fit.")
        _validate_feature_matrix(features)

        predictions: list[float] = []
        for row in features:
            design_row = [1.0] + self._standardize(row)
            prediction = sum(weight * value for weight, value in zip(self.coefficients, design_row))
            if self.non_negative:
                prediction = max(0.0, prediction)
            predictions.append(prediction)
        return predictions

    def to_dict(self) -> dict:
        if self.coefficients is None:
            raise ValueError("Model has not been fit.")
        return {
            "type": "RidgeRegressor",
            "alpha": self.alpha,
            "non_negative": self.non_negative,
            "coefficients": self.coefficients,
            "means": self.means,
            "scales": self.scales,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RidgeRegressor":
        model = cls(alpha=data["alpha"], non_negative=data["non_negative"])
        model.coefficients = data["coefficients"]
        model.means = data["means"]
        model.scales = data["scales"]
        return model

    def _standardize(self, row: list[float]) -> list[float]:
        if self.means is None or self.scales is None:
            raise ValueError("Standardization statistics are not available.")
        if len(row) != len(self.means):
            raise ValueError(
                f"Expected {len(self.means)} features, received {len(row)}."
            )
        return [(value - mean) / scale for value, mean, scale in zip(row, self.means, self.scales)]


@dataclass
class SeasonalNaiveModel:
    """Historical day-of-year mean baseline."""

    target_column: str
    fallback: float = 0.0
    daily_means: dict[int, float] | None = None

    def fit(self, records: list[WeatherRecord]) -> "SeasonalNaiveModel":
        if not records:
            raise ValueError("Cannot fit seasonal baseline on an empty record set.")

        values_by_day: dict[int, list[float]] = defaultdict(list)
        all_values: list[float] = []
        for record in records:
            value = record.values[self.target_column]
            day = min(record.date.timetuple().tm_yday, 365)
            values_by_day[day].append(value)
            all_values.append(value)

        self.fallback = sum(all_values) / len(all_values)
        self.daily_means = {
            day: sum(values) / len(values) for day, values in values_by_day.items()
        }
        return self

    def predict(self, dates: list[date]) -> list[float]:
        if self.daily_means is None:
            raise ValueError("Model has not been fit.")
        return [
            self.daily_means.get(min(day.timetuple().tm_yday, 365), self.fallback)
            for day in dates
        ]

    def to_dict(self) -> dict:
        if self.daily_means is None:
            raise ValueError("Model has not been fit.")
        return {
            "type": "SeasonalNaiveModel",
            "target_column": self.target_column,
            "fallback": self.fallback,
            "daily_means": {str(k): v for k, v in self.daily_means.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SeasonalNaiveModel":
        model = cls(target_column=data["target_column"], fallback=data["fallback"])
        model.daily_means = {int(k): v for k, v in data["daily_means"].items()}
        return model


def _validate_feature_matrix(features: list[list[float]]) -> None:
    if not features:
        raise ValueError("Feature matrix must contain at least one row.")

    width = len(features[0])
    if width == 0:
        raise ValueError("Feature matrix must contain at least one column.")

    for row in features:
        if len(row) != width:
            raise ValueError("All feature rows must have the same width.")


def _column_stats(features: list[list[float]]) -> tuple[list[float], list[float]]:
    width = len(features[0])
    means: list[float] = []
    scales: list[float] = []
    for column_index in range(width):
        column = [row[column_index] for row in features]
        column_mean = sum(column) / len(column)
        variance = sum((value - column_mean) ** 2 for value in column) / len(column)
        means.append(column_mean)
        scales.append(sqrt(variance) or 1.0)
    return means, scales


def _normal_equations(
    design: list[list[float]],
    target: list[float],
) -> tuple[list[list[float]], list[float]]:
    width = len(design[0])
    xtx = [[0.0 for _ in range(width)] for _ in range(width)]
    xty = [0.0 for _ in range(width)]

    for row, y_value in zip(design, target):
        for i in range(width):
            xty[i] += row[i] * y_value
            for j in range(width):
                xtx[i][j] += row[i] * row[j]

    return xtx, xty


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve Ax=b with Gauss-Jordan elimination and partial pivoting."""

    n = len(vector)
    augmented = [row[:] + [rhs] for row, rhs in zip(matrix, vector)]

    for column in range(n):
        pivot_row = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot_row][column]) < 1e-12:
            raise ValueError("Singular matrix while fitting ridge regression.")
        augmented[column], augmented[pivot_row] = augmented[pivot_row], augmented[column]

        pivot = augmented[column][column]
        augmented[column] = [value / pivot for value in augmented[column]]

        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]

    return [row[-1] for row in augmented]

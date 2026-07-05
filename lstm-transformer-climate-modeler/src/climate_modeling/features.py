"""Feature engineering for one-day-ahead weather modeling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from climate_modeling.data import TARGET_COLUMNS, WeatherRecord


@dataclass(frozen=True)
class SupervisedDataset:
    """Tabular supervised-learning data."""

    features: list[list[float]]
    target: list[float]
    dates: list[date]
    feature_names: list[str]


def build_supervised_dataset(
    records: list[WeatherRecord],
    target_column: str,
    start: date,
    end: date,
    min_history_days: int = 30,
) -> SupervisedDataset:
    """Build lagged and seasonal features for a target variable.

    Features for day ``t`` use calendar signals and observations from days
    before ``t``. The current day's target is never included as an input.
    """

    if target_column not in TARGET_COLUMNS:
        raise ValueError(f"Unsupported target {target_column!r}.")
    if len(records) <= min_history_days:
        raise ValueError("Not enough records to build lag features.")

    records = sorted(records, key=lambda record: record.date)
    feature_names = _feature_names()
    rows: list[list[float]] = []
    target: list[float] = []
    dates: list[date] = []

    first_date = records[0].date
    for index in range(min_history_days, len(records)):
        record = records[index]
        if not (start <= record.date <= end):
            continue

        rows.append(_features_for_index(records, index, first_date))
        target.append(record.values[target_column])
        dates.append(record.date)

    if not rows:
        raise ValueError(f"No supervised rows for {target_column} in requested window.")

    return SupervisedDataset(rows, target, dates, feature_names)


def _feature_names() -> list[str]:
    names = [
        "day_sin",
        "day_cos",
        "semiannual_sin",
        "semiannual_cos",
        "trend_years",
    ]
    for column in TARGET_COLUMNS:
        names.extend(
            [
                f"{column}_lag_1",
                f"{column}_lag_7",
                f"{column}_lag_30",
                f"{column}_rolling_7",
                f"{column}_rolling_30",
            ]
        )
    return names


def _features_for_index(records: list[WeatherRecord], index: int, first_date: date) -> list[float]:
    current = records[index]
    day_angle = 2.0 * math.pi * (current.date.timetuple().tm_yday - 1) / 365.25
    semiannual_angle = 2.0 * day_angle
    trend_years = (current.date - first_date).days / 365.25

    features = [
        math.sin(day_angle),
        math.cos(day_angle),
        math.sin(semiannual_angle),
        math.cos(semiannual_angle),
        trend_years,
    ]

    for column in TARGET_COLUMNS:
        series = [record.values[column] for record in records]
        features.extend(
            [
                series[index - 1],
                series[index - 7],
                series[index - 30],
                _mean(series[index - 7 : index]),
                _mean(series[index - 30 : index]),
            ]
        )

    return features


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)

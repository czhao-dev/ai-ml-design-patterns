"""Sliding-window sequence dataset builder for deep-learning weather forecasting.

Each sample maps a ``lookback``-day window of observations to the following
``horizon`` days of [PRCP, SNOW, TOBS] — enabling multi-step forecasting.
The :class:`SequenceScaler` normalises features and targets using training
statistics only, preventing any leakage from the test period.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from climate_modeling.data import WeatherRecord

# ---------------------------------------------------------------------------
# Feature / target layout
# ---------------------------------------------------------------------------

SEQUENCE_FEATURE_NAMES: tuple[str, ...] = (
    "day_sin",        # annual cycle  – sin component
    "day_cos",        # annual cycle  – cos component
    "semiannual_sin", # 6-month cycle – sin component
    "semiannual_cos", # 6-month cycle – cos component
    "trend_years",    # long-run linear trend
    "TOBS",           # observed temperature (°F)
    "PRCP",           # precipitation (in)
    "SNOW",           # snowfall (in)
    "SNWD",           # snow depth (in)
)
N_FEATURES: int = len(SEQUENCE_FEATURE_NAMES)  # 9

# Prediction targets – order used throughout tf_models (PRCP=0, SNOW=1, TOBS=2)
SEQ_TARGET_NAMES: tuple[str, ...] = ("PRCP", "SNOW", "TOBS")
N_TARGETS: int = len(SEQ_TARGET_NAMES)  # 3


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------


@dataclass
class SequenceScaler:
    """Per-feature / per-target z-score normaliser fitted on training data.

    All arrays are expected to be ``float32``.
    """

    feature_means: list[float]
    feature_stds: list[float]
    target_means: list[float]
    target_stds: list[float]

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray) -> "SequenceScaler":
        """Compute per-feature statistics from raw training arrays.

        Args:
            X: shape ``(n_windows, lookback, N_FEATURES)``
            y: shape ``(n_windows, horizon, N_TARGETS)``
        """
        eps = 1e-8
        flat_X = X.reshape(-1, X.shape[-1])
        flat_y = y.reshape(-1, y.shape[-1])
        return cls(
            feature_means=flat_X.mean(axis=0).tolist(),
            feature_stds=(flat_X.std(axis=0) + eps).tolist(),
            target_means=flat_y.mean(axis=0).tolist(),
            target_stds=(flat_y.std(axis=0) + eps).tolist(),
        )

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Z-score normalise feature windows using training statistics."""
        means = np.array(self.feature_means, dtype=np.float32)
        stds = np.array(self.feature_stds, dtype=np.float32)
        return ((X - means) / stds).astype(np.float32)

    def transform_targets(self, y: np.ndarray) -> np.ndarray:
        """Z-score normalise target windows using training statistics."""
        means = np.array(self.target_means, dtype=np.float32)
        stds = np.array(self.target_stds, dtype=np.float32)
        return ((y - means) / stds).astype(np.float32)

    def inverse_transform_targets(self, y: np.ndarray) -> np.ndarray:
        """Undo target normalisation, returning values in original units."""
        means = np.array(self.target_means, dtype=np.float32)
        stds = np.array(self.target_stds, dtype=np.float32)
        return (y * stds + means).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "target_means": self.target_means,
            "target_stds": self.target_stds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SequenceScaler":
        return cls(**data)


# ---------------------------------------------------------------------------
# Public dataset builder
# ---------------------------------------------------------------------------


def build_sequence_dataset(
    records: list[WeatherRecord],
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    lookback: int = 60,
    horizon: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SequenceScaler]:
    """Build sliding-window datasets for LSTM / Transformer training.

    For each day *d* in ``[train_start, train_end]`` (and separately
    ``[test_start, test_end]``) a window is created whose:

    * **input**  spans days ``[d − lookback + 1, d]``       → shape ``(lookback, N_FEATURES)``
    * **target** spans days ``[d + 1,   d + horizon]``      → shape ``(horizon,  N_TARGETS)``

    Windows without sufficient history (``d − lookback + 1 < 0``) or
    insufficient future data (``d + horizon ≥ len(records)``) are skipped.

    The :class:`SequenceScaler` is **fitted only on training windows** and
    then applied to both splits.

    Returns:
        X_train: ``(n_train, lookback, N_FEATURES)`` – normalised
        y_train: ``(n_train, horizon,  N_TARGETS)``  – normalised
        X_test:  ``(n_test,  lookback, N_FEATURES)`` – normalised
        y_test:  ``(n_test,  horizon,  N_TARGETS)``  – normalised
        scaler:  :class:`SequenceScaler` fitted on training data
    """
    records = sorted(records, key=lambda r: r.date)
    first_date = records[0].date

    daily_features = _build_daily_features(records, first_date)
    daily_targets = _build_daily_targets(records)
    dates = [r.date for r in records]
    date_to_idx: dict[date, int] = {d: i for i, d in enumerate(dates)}

    X_raw_train, y_raw_train = _extract_windows(
        daily_features, daily_targets, dates, date_to_idx,
        train_start, train_end, lookback, horizon,
    )
    X_raw_test, y_raw_test = _extract_windows(
        daily_features, daily_targets, dates, date_to_idx,
        test_start, test_end, lookback, horizon,
    )

    scaler = SequenceScaler.fit(X_raw_train, y_raw_train)

    return (
        scaler.transform_features(X_raw_train),
        scaler.transform_targets(y_raw_train),
        scaler.transform_features(X_raw_test),
        scaler.transform_targets(y_raw_test),
        scaler,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_daily_features(
    records: list[WeatherRecord], first_date: date
) -> np.ndarray:
    """Return raw (un-normalised) feature matrix of shape ``(n_days, N_FEATURES)``."""
    n = len(records)
    out = np.zeros((n, N_FEATURES), dtype=np.float32)
    for i, rec in enumerate(records):
        yday = rec.date.timetuple().tm_yday
        day_angle = 2.0 * math.pi * (yday - 1) / 365.25
        sa_angle = 2.0 * day_angle
        trend = (rec.date - first_date).days / 365.25
        out[i] = (
            math.sin(day_angle),
            math.cos(day_angle),
            math.sin(sa_angle),
            math.cos(sa_angle),
            trend,
            rec.values["TOBS"],
            rec.values["PRCP"],
            rec.values["SNOW"],
            rec.values["SNWD"],
        )
    return out


def _build_daily_targets(records: list[WeatherRecord]) -> np.ndarray:
    """Return raw target matrix of shape ``(n_days, N_TARGETS)``."""
    n = len(records)
    out = np.zeros((n, N_TARGETS), dtype=np.float32)
    for i, rec in enumerate(records):
        out[i] = (
            rec.values["PRCP"],
            rec.values["SNOW"],
            rec.values["TOBS"],
        )
    return out


def _extract_windows(
    features: np.ndarray,
    targets: np.ndarray,
    dates: list[date],
    date_to_idx: dict[date, int],
    start: date,
    end: date,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect sliding windows whose last input day falls in ``[start, end]``."""
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for d in dates:
        if not (start <= d <= end):
            continue
        end_idx = date_to_idx[d]
        start_idx = end_idx - lookback + 1
        target_end_idx = end_idx + horizon

        if start_idx < 0 or target_end_idx >= len(dates):
            continue

        X_list.append(features[start_idx : end_idx + 1])           # (lookback, N_FEATURES)
        y_list.append(targets[end_idx + 1 : target_end_idx + 1])   # (horizon,  N_TARGETS)

    if not X_list:
        raise ValueError(
            f"No sequence windows found for [{start}, {end}]. "
            f"Records must extend at least {lookback} days before {start} "
            f"and {horizon} days after {end}."
        )

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

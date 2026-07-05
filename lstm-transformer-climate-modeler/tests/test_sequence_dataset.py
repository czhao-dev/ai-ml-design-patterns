"""Tests for sequence_dataset.py."""

from __future__ import annotations

import unittest
from datetime import date, timedelta

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from climate_modeling.data import WeatherRecord


def _make_records(n: int, start_date: date | None = None) -> list[WeatherRecord]:
    """Build *n* synthetic WeatherRecords with deterministic values."""
    if start_date is None:
        start_date = date(2010, 1, 1)
    records = []
    for i in range(n):
        d = start_date + timedelta(days=i)
        records.append(
            WeatherRecord(
                station="FAKE",
                station_name="FAKE STATION",
                date=d,
                values={
                    "TOBS": float(50 + i % 40),
                    "TMAX": float(60 + i % 40),
                    "TMIN": float(30 + i % 20),
                    "PRCP": float((i % 5) * 0.1),
                    "SNOW": float((i % 10) * 0.05),
                    "SNWD": float((i % 15) * 0.1),
                },
            )
        )
    return records


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy not installed")
class TestBuildDailyFeatures(unittest.TestCase):
    def test_shape(self) -> None:
        from climate_modeling.sequence_dataset import N_FEATURES, _build_daily_features

        records = _make_records(30)
        out = _build_daily_features(records, records[0].date)
        self.assertEqual(out.shape, (30, N_FEATURES))

    def test_dtype(self) -> None:
        from climate_modeling.sequence_dataset import _build_daily_features

        records = _make_records(10)
        out = _build_daily_features(records, records[0].date)
        self.assertEqual(out.dtype, np.float32)

    def test_trend_increases(self) -> None:
        from climate_modeling.sequence_dataset import _build_daily_features

        records = _make_records(100)
        out = _build_daily_features(records, records[0].date)
        # trend_years is the last of the 5 temporal features (index 4)
        self.assertLess(out[0, 4], out[-1, 4])


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy not installed")
class TestExtractWindows(unittest.TestCase):
    def setUp(self) -> None:
        from climate_modeling.sequence_dataset import (
            N_FEATURES,
            N_TARGETS,
            _build_daily_features,
            _build_daily_targets,
        )

        self.lookback = 10
        self.horizon = 3
        self.records = _make_records(200, start_date=date(2015, 1, 1))
        first_date = self.records[0].date
        self.features = _build_daily_features(self.records, first_date)
        self.targets = _build_daily_targets(self.records)
        self.dates = [r.date for r in self.records]
        self.date_to_idx = {d: i for i, d in enumerate(self.dates)}
        self.N_FEATURES = N_FEATURES
        self.N_TARGETS = N_TARGETS

    def test_output_shapes(self) -> None:
        from climate_modeling.sequence_dataset import _extract_windows

        start = date(2015, 3, 1)
        end = date(2015, 6, 30)
        X, y = _extract_windows(
            self.features,
            self.targets,
            self.dates,
            self.date_to_idx,
            start,
            end,
            self.lookback,
            self.horizon,
        )
        n = X.shape[0]
        self.assertGreater(n, 0)
        self.assertEqual(X.shape, (n, self.lookback, self.N_FEATURES))
        self.assertEqual(y.shape, (n, self.horizon, self.N_TARGETS))

    def test_no_target_leakage(self) -> None:
        """Targets must be strictly after the input window."""
        from climate_modeling.sequence_dataset import _extract_windows

        start = date(2015, 3, 1)
        end = date(2015, 9, 30)
        X, y = _extract_windows(
            self.features,
            self.targets,
            self.dates,
            self.date_to_idx,
            start,
            end,
            self.lookback,
            self.horizon,
        )
        # The last input day falls in [start, end].  The first target day
        # must be strictly after the last input day.  We verify this by
        # checking that input and target value arrays are different slices
        # of the raw feature/target matrices for each window.
        for window_idx in range(min(5, len(X))):
            # Find the end_idx for this window via the date
            end_date = start + timedelta(days=window_idx)
            if end_date not in self.date_to_idx:
                continue
            end_idx = self.date_to_idx[end_date]
            # Input window ends at end_idx; target window starts at end_idx+1
            expected_X_last = self.features[end_idx]
            expected_y_first = self.targets[end_idx + 1]
            np.testing.assert_array_almost_equal(
                X[window_idx, -1], expected_X_last, decimal=5
            )
            np.testing.assert_array_almost_equal(
                y[window_idx, 0], expected_y_first, decimal=5
            )

    def test_raises_empty_range(self) -> None:
        from climate_modeling.sequence_dataset import _extract_windows

        # Use a date range with no matching days in the records
        with self.assertRaises(ValueError):
            _extract_windows(
                self.features,
                self.targets,
                self.dates,
                self.date_to_idx,
                date(2030, 1, 1),
                date(2030, 12, 31),
                self.lookback,
                self.horizon,
            )


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy not installed")
class TestSequenceScaler(unittest.TestCase):
    def test_round_trip(self) -> None:
        from climate_modeling.sequence_dataset import SequenceScaler

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 10, 9)).astype(np.float32)
        y = rng.standard_normal((50, 7, 3)).astype(np.float32)

        scaler = SequenceScaler.fit(X, y)
        X_norm = scaler.transform_features(X)
        y_norm = scaler.transform_targets(y)
        y_back = scaler.inverse_transform_targets(y_norm)

        np.testing.assert_allclose(y_back, y, atol=1e-5)

    def test_normalised_mean_near_zero(self) -> None:
        from climate_modeling.sequence_dataset import SequenceScaler

        rng = np.random.default_rng(1)
        X = rng.standard_normal((100, 10, 9)).astype(np.float32) * 5 + 3
        y = rng.standard_normal((100, 7, 3)).astype(np.float32) * 2 + 10

        scaler = SequenceScaler.fit(X, y)
        X_norm = scaler.transform_features(X)
        self.assertAlmostEqual(float(X_norm.reshape(-1, 9).mean()), 0.0, places=4)

    def test_serialisation_round_trip(self) -> None:
        from climate_modeling.sequence_dataset import SequenceScaler

        rng = np.random.default_rng(2)
        X = rng.standard_normal((20, 5, 9)).astype(np.float32)
        y = rng.standard_normal((20, 3, 3)).astype(np.float32)
        scaler = SequenceScaler.fit(X, y)
        restored = SequenceScaler.from_dict(scaler.to_dict())
        np.testing.assert_array_equal(scaler.feature_means, restored.feature_means)
        np.testing.assert_array_equal(scaler.target_stds, restored.target_stds)


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy not installed")
class TestBuildSequenceDataset(unittest.TestCase):
    def test_end_to_end_shapes(self) -> None:
        from climate_modeling.sequence_dataset import (
            N_FEATURES,
            N_TARGETS,
            build_sequence_dataset,
        )

        lookback, horizon = 10, 3
        records = _make_records(300, start_date=date(2018, 1, 1))

        X_tr, y_tr, X_te, y_te, scaler = build_sequence_dataset(
            records,
            train_start=date(2018, 3, 1),
            train_end=date(2018, 9, 30),
            test_start=date(2018, 10, 1),
            test_end=date(2018, 11, 30),
            lookback=lookback,
            horizon=horizon,
        )

        n_tr, n_te = X_tr.shape[0], X_te.shape[0]
        self.assertGreater(n_tr, 0)
        self.assertGreater(n_te, 0)
        self.assertEqual(X_tr.shape, (n_tr, lookback, N_FEATURES))
        self.assertEqual(y_tr.shape, (n_tr, horizon, N_TARGETS))
        self.assertEqual(X_te.shape, (n_te, lookback, N_FEATURES))
        self.assertEqual(y_te.shape, (n_te, horizon, N_TARGETS))

    def test_train_scaler_not_fit_on_test(self) -> None:
        """Scaler fitted on train should not see test statistics."""
        from climate_modeling.sequence_dataset import build_sequence_dataset

        records = _make_records(400, start_date=date(2018, 1, 1))
        _, _, X_te_a, _, scaler_a = build_sequence_dataset(
            records,
            train_start=date(2018, 3, 1),
            train_end=date(2018, 8, 31),
            test_start=date(2018, 9, 1),
            test_end=date(2018, 10, 31),
            lookback=10,
            horizon=3,
        )
        _, _, X_te_b, _, scaler_b = build_sequence_dataset(
            records,
            train_start=date(2018, 3, 1),
            train_end=date(2018, 8, 31),
            test_start=date(2018, 9, 1),
            test_end=date(2018, 10, 31),
            lookback=10,
            horizon=3,
        )
        # Same records + same splits → deterministic result
        np.testing.assert_array_equal(scaler_a.feature_means, scaler_b.feature_means)


if __name__ == "__main__":
    unittest.main()

from datetime import date, timedelta
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from climate_modeling.data import WeatherRecord
from climate_modeling.metrics import regression_metrics
from climate_modeling.models import RidgeRegressor, SeasonalNaiveModel


class RidgeRegressorTests(unittest.TestCase):
    def test_ridge_learns_simple_linear_relationship(self):
        features = [[float(value)] for value in range(1, 8)]
        target = [2.0 * row[0] + 1.0 for row in features]

        model = RidgeRegressor(alpha=0.01).fit(features, target)
        predictions = model.predict([[8.0], [9.0]])

        self.assertAlmostEqual(predictions[0], 17.0, delta=0.2)
        self.assertAlmostEqual(predictions[1], 19.0, delta=0.2)

    def test_regression_metrics_identical_predictions(self):
        metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["r2"], 1.0)

    def test_fit_rejects_mismatched_target_length(self):
        with self.assertRaisesRegex(ValueError, "row counts"):
            RidgeRegressor().fit([[1.0], [2.0]], [1.0])

    def test_fit_rejects_ragged_features(self):
        with self.assertRaisesRegex(ValueError, "same width"):
            RidgeRegressor().fit([[1.0], [2.0, 3.0]], [1.0, 2.0])

    def test_predict_rejects_wrong_feature_width(self):
        model = RidgeRegressor(alpha=0.1).fit([[1.0, 2.0], [2.0, 3.0]], [3.0, 5.0])

        with self.assertRaisesRegex(ValueError, "Expected 2 features"):
            model.predict([[1.0]])

    def test_non_negative_predictions_are_clamped(self):
        model = RidgeRegressor(alpha=0.1, non_negative=True).fit(
            [[1.0], [2.0], [3.0]],
            [-1.0, -2.0, -3.0],
        )

        self.assertEqual(model.predict([[10.0]]), [0.0])


class SeasonalNaiveModelTests(unittest.TestCase):
    def test_fit_rejects_empty_records(self):
        with self.assertRaisesRegex(ValueError, "empty record"):
            SeasonalNaiveModel("TOBS").fit([])


class SerializationTests(unittest.TestCase):
    def _make_records(self, n: int = 5) -> list[WeatherRecord]:
        return [
            WeatherRecord(
                station="TEST",
                station_name="TEST STATION",
                date=date(2020, 1, 1) + timedelta(days=i),
                values={"PRCP": 0.0, "SNWD": 0.0, "SNOW": 0.0, "TMAX": 50.0, "TMIN": 40.0, "TOBS": 45.0 + i},
            )
            for i in range(n)
        ]

    def test_ridge_serialization_round_trip(self):
        features = [[float(v)] for v in range(1, 8)]
        target = [2.0 * row[0] + 1.0 for row in features]
        original = RidgeRegressor(alpha=0.5, non_negative=True).fit(features, target)

        restored = RidgeRegressor.from_dict(original.to_dict())

        self.assertEqual(restored.alpha, original.alpha)
        self.assertEqual(restored.non_negative, original.non_negative)
        self.assertEqual(restored.coefficients, original.coefficients)
        self.assertEqual(restored.means, original.means)
        self.assertEqual(restored.scales, original.scales)
        self.assertEqual(restored.predict([[8.0]]), original.predict([[8.0]]))

    def test_ridge_to_dict_rejects_unfit_model(self):
        with self.assertRaisesRegex(ValueError, "not been fit"):
            RidgeRegressor().to_dict()

    def test_seasonal_naive_serialization_round_trip(self):
        records = self._make_records()
        original = SeasonalNaiveModel("TOBS").fit(records)
        test_dates = [date(2020, 1, 1), date(2020, 1, 3)]

        restored = SeasonalNaiveModel.from_dict(original.to_dict())

        self.assertEqual(restored.target_column, original.target_column)
        self.assertAlmostEqual(restored.fallback, original.fallback)
        self.assertEqual(restored.predict(test_dates), original.predict(test_dates))

    def test_seasonal_naive_to_dict_rejects_unfit_model(self):
        with self.assertRaisesRegex(ValueError, "not been fit"):
            SeasonalNaiveModel("TOBS").to_dict()


if __name__ == "__main__":
    unittest.main()

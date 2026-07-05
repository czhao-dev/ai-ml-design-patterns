from datetime import date, timedelta
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from climate_modeling.data import WeatherRecord, train_test_split
from climate_modeling.features import build_supervised_dataset


def make_record(day_index: int) -> WeatherRecord:
    day = date(2020, 1, 1) + timedelta(days=day_index)
    return WeatherRecord(
        station="TEST",
        station_name="TEST STATION",
        date=day,
        values={
            "PRCP": float(day_index % 3),
            "SNWD": 0.0,
            "SNOW": float(day_index % 2),
            "TMAX": 50.0 + day_index,
            "TMIN": 40.0 + day_index,
            "TOBS": 45.0 + day_index,
        },
    )


class PipelineTests(unittest.TestCase):
    def test_train_test_split_uses_inclusive_dates(self):
        records = [make_record(index) for index in range(10)]

        train, test = train_test_split(
            records,
            train_start=date(2020, 1, 2),
            train_end=date(2020, 1, 4),
            test_start=date(2020, 1, 8),
            test_end=date(2020, 1, 9),
        )

        self.assertEqual([record.date for record in train], [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)])
        self.assertEqual([record.date for record in test], [date(2020, 1, 8), date(2020, 1, 9)])

    def test_supervised_dataset_uses_history_only(self):
        records = [make_record(index) for index in range(40)]
        dataset = build_supervised_dataset(
            records,
            target_column="TOBS",
            start=date(2020, 1, 31),
            end=date(2020, 2, 1),
            min_history_days=30,
        )

        self.assertEqual(len(dataset.features), 2)
        self.assertIn("TOBS_lag_1", dataset.feature_names)
        lag_index = dataset.feature_names.index("TOBS_lag_1")
        self.assertEqual(dataset.features[0][lag_index], records[29].values["TOBS"])
        self.assertEqual(dataset.target[0], records[30].values["TOBS"])

    def test_supervised_dataset_sorts_records_before_building_lags(self):
        records = [make_record(index) for index in range(40)]
        shuffled = list(reversed(records))

        dataset = build_supervised_dataset(
            shuffled,
            target_column="TOBS",
            start=date(2020, 1, 31),
            end=date(2020, 1, 31),
            min_history_days=30,
        )

        lag_index = dataset.feature_names.index("TOBS_lag_1")
        self.assertEqual(dataset.features[0][lag_index], records[29].values["TOBS"])

    def test_supervised_dataset_rejects_unknown_target(self):
        with self.assertRaisesRegex(ValueError, "Unsupported target"):
            build_supervised_dataset(
                [make_record(index) for index in range(40)],
                target_column="TMAX",
                start=date(2020, 1, 31),
                end=date(2020, 1, 31),
            )


if __name__ == "__main__":
    unittest.main()

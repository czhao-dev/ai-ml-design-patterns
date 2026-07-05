from pathlib import Path
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from climate_modeling.data import load_station_records, parse_iso_date


CSV_HEADER = (
    "STATION,STATION_NAME,DATE,PRCP,SNWD,SNOW,TMAX,TMIN,TOBS\n"
)


class DataLoadingTests(unittest.TestCase):
    def test_load_station_records_cleans_missing_values_and_sorts(self):
        content = CSV_HEADER + textwrap.dedent(
            """\
            TEST,OTHER STATION,20200101,99,99,99,99,99,99
            TEST,READING MA US,20200102,-9999,-9999,-9999,42,30,-9999
            TEST,READING MA US,20200101,0.5,1,2,-9999,20,30
            TEST,READING MA US,20200103,0,0,0,-9999,-9999,-9999
            """
        )

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "weather.csv"
            csv_path.write_text(content, encoding="utf-8")

            records = load_station_records(csv_path)

        self.assertEqual([record.date.isoformat() for record in records], ["2020-01-01", "2020-01-02", "2020-01-03"])
        self.assertEqual(records[1].values["PRCP"], 0.0)
        self.assertEqual(records[1].values["SNWD"], 0.0)
        self.assertEqual(records[1].values["SNOW"], 0.0)
        self.assertEqual(records[0].values["TMAX"], 40.0)
        self.assertEqual(records[1].values["TOBS"], 36.0)
        self.assertEqual(records[2].values["TOBS"], 33.0)

    def test_load_station_records_rejects_missing_station(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "weather.csv"
            csv_path.write_text(CSV_HEADER, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "No rows found"):
                load_station_records(csv_path, station_name="MISSING")

    def test_load_station_records_rejects_unfillable_temperature_columns(self):
        content = CSV_HEADER + "TEST,READING MA US,20200101,0,0,0,-9999,-9999,-9999\n"

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "weather.csv"
            csv_path.write_text(content, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no observed values"):
                load_station_records(csv_path)

    def test_parse_iso_date_rejects_non_iso_format(self):
        with self.assertRaises(ValueError):
            parse_iso_date("01/01/2020")


if __name__ == "__main__":
    unittest.main()

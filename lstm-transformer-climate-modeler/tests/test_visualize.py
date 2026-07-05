from datetime import date
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from climate_modeling.visualize import write_actual_vs_predicted_svg


class VisualizeTests(unittest.TestCase):
    def test_write_actual_vs_predicted_svg_creates_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "chart.svg"

            write_actual_vs_predicted_svg(
                output_path,
                [date(2020, 1, 1), date(2020, 1, 2)],
                [1.0, 2.0],
                [1.1, 1.9],
                title="Test chart",
                ylabel="Units",
            )

            content = output_path.read_text(encoding="utf-8")

        self.assertIn("<svg", content)
        self.assertIn("Test chart", content)
        self.assertIn("Predicted", content)

    def test_write_actual_vs_predicted_svg_rejects_mismatched_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "equal length"):
                write_actual_vs_predicted_svg(
                    Path(directory) / "chart.svg",
                    [date(2020, 1, 1)],
                    [1.0, 2.0],
                    [1.0],
                    title="Bad chart",
                    ylabel="Units",
                )

    def test_write_actual_vs_predicted_svg_rejects_empty_series(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "empty date"):
                write_actual_vs_predicted_svg(
                    Path(directory) / "chart.svg",
                    [],
                    [],
                    [],
                    title="Bad chart",
                    ylabel="Units",
                )


if __name__ == "__main__":
    unittest.main()

"""Command-line training workflow for the climate modeling project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from climate_modeling.data import (
    TARGET_COLUMNS,
    load_station_records,
    parse_iso_date,
    train_test_split,
)
from climate_modeling.features import build_supervised_dataset
from climate_modeling.metrics import regression_metrics
from climate_modeling.models import RidgeRegressor, SeasonalNaiveModel
from climate_modeling.visualize import write_actual_vs_predicted_svg


TARGET_LABELS = {
    "PRCP": "Precipitation (inches)",
    "SNOW": "Snowfall (inches)",
    "TOBS": "Observed temperature (F)",
}


def main() -> None:
    args = _parse_args()
    reports_dir = Path(args.reports_dir)
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_start = parse_iso_date(args.train_start)
    train_end = parse_iso_date(args.train_end)
    test_start = parse_iso_date(args.test_start)
    test_end = parse_iso_date(args.test_end)

    records = load_station_records(args.data, args.station)
    train_records, test_records = train_test_split(
        records,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    saved_models: dict[str, dict] = {}

    report = {
        "project": "ML Climate Modeling",
        "station": args.station,
        "data_path": str(args.data),
        "train_window": [args.train_start, args.train_end],
        "test_window": [args.test_start, args.test_end],
        "train_rows": len(train_records),
        "test_rows": len(test_records),
        "models": {},
    }

    for target in TARGET_COLUMNS:
        train_dataset = build_supervised_dataset(records, target, train_start, train_end)
        test_dataset = build_supervised_dataset(records, target, test_start, test_end)

        baseline = SeasonalNaiveModel(target).fit(train_records)
        baseline_predictions = baseline.predict(test_dataset.dates)

        model = RidgeRegressor(
            alpha=args.alpha,
            non_negative=target in {"PRCP", "SNOW"},
        ).fit(train_dataset.features, train_dataset.target)
        predictions = model.predict(test_dataset.features)

        saved_models[target] = {"ridge": model.to_dict(), "baseline": baseline.to_dict()}

        report["models"][target] = {
            "target_label": TARGET_LABELS[target],
            "feature_count": len(train_dataset.feature_names),
            "features": train_dataset.feature_names,
            "baseline": regression_metrics(test_dataset.target, baseline_predictions),
            "ridge_regression": regression_metrics(test_dataset.target, predictions),
        }

        write_actual_vs_predicted_svg(
            figures_dir / f"{target.lower()}_actual_vs_predicted.svg",
            test_dataset.dates,
            test_dataset.target,
            predictions,
            title=f"{target}: {args.test_start} to {args.test_end} actual vs ridge forecast",
            ylabel=TARGET_LABELS[target],
        )

    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    models_path = reports_dir / "models.json"
    models_path.write_text(json.dumps(saved_models, indent=2), encoding="utf-8")

    _print_summary(report, metrics_path, models_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train weather forecasting models.")
    parser.add_argument("--data", default="962598.csv", help="Path to the NOAA CSV export.")
    parser.add_argument("--station", default="READING MA US", help="Station name to model.")
    parser.add_argument("--train-start", default="2013-01-01")
    parser.add_argument("--train-end", default="2015-12-31")
    parser.add_argument("--test-start", default="2016-01-01")
    parser.add_argument("--test-end", default="2016-12-31")
    parser.add_argument("--alpha", type=float, default=10.0, help="Ridge regularization strength.")
    parser.add_argument("--reports-dir", default="reports", help="Directory for metrics and figures.")
    return parser.parse_args()


def load_models(models_path: str | Path) -> dict[str, dict]:
    """Load serialized models from a JSON file produced by the training pipeline.

    Returns a dict keyed by target column (e.g. ``"TOBS"``), each containing
    ``"ridge"`` (a fitted :class:`RidgeRegressor`) and ``"baseline"`` (a fitted
    :class:`SeasonalNaiveModel`).

    Example::

        from climate_modeling.train import load_models
        from climate_modeling.features import build_supervised_dataset

        models = load_models("reports/models.json")
        dataset = build_supervised_dataset(records, "TOBS", start, end)
        predictions = models["TOBS"]["ridge"].predict(dataset.features)
    """
    data = json.loads(Path(models_path).read_text(encoding="utf-8"))
    return {
        target: {
            "ridge": RidgeRegressor.from_dict(entry["ridge"]),
            "baseline": SeasonalNaiveModel.from_dict(entry["baseline"]),
        }
        for target, entry in data.items()
    }


def _print_summary(report: dict, metrics_path: Path, models_path: Path) -> None:
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved models  to {models_path}")
    print(f"Station: {report['station']}")
    print(f"Train rows: {report['train_rows']} | Test rows: {report['test_rows']}")
    print("")
    print("Target  Baseline RMSE  Ridge RMSE  Ridge R2")
    print("------  -------------  ----------  --------")
    for target, results in report["models"].items():
        baseline_rmse = results["baseline"]["rmse"]
        ridge_rmse = results["ridge_regression"]["rmse"]
        ridge_r2 = results["ridge_regression"]["r2"]
        print(f"{target:<6}  {baseline_rmse:>13.3f}  {ridge_rmse:>10.3f}  {ridge_r2:>8.3f}")


if __name__ == "__main__":
    main()

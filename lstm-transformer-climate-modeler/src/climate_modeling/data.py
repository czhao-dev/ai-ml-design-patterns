"""Data loading and cleaning for NOAA daily station exports."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from statistics import mean


WEATHER_COLUMNS = ("PRCP", "SNWD", "SNOW", "TMAX", "TMIN", "TOBS")
TARGET_COLUMNS = ("PRCP", "SNOW", "TOBS")
MISSING_VALUE = -9999.0


@dataclass(frozen=True)
class WeatherRecord:
    """Clean daily weather observation for one station."""

    station: str
    station_name: str
    date: date
    values: dict[str, float]


def load_station_records(
    csv_path: str | Path,
    station_name: str = "READING MA US",
) -> list[WeatherRecord]:
    """Load, sort, and clean one station from a NOAA CSV export.

    The source file uses ``-9999`` for missing observations. Precipitation,
    snow depth, and snowfall are treated as zero when missing; temperatures are
    reconstructed from the available daily high/low/observed values when
    possible and otherwise filled with same-month/day climatology.
    """

    path = Path(csv_path)
    raw_records: list[WeatherRecord] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        # Support both old CDO format (STATION_NAME) and new CDO format (NAME)
        name_col = "NAME" if "NAME" in (reader.fieldnames or []) else "STATION_NAME"
        for row in reader:
            if row[name_col].strip() != station_name:
                continue

            values = {column: _parse_float(row.get(column)) for column in WEATHER_COLUMNS}
            raw_records.append(
                WeatherRecord(
                    station=row["STATION"].strip(),
                    station_name=row[name_col].strip(),
                    date=_parse_date(row["DATE"]),
                    values=_clean_precipitation(values),
                )
            )

    if not raw_records:
        raise ValueError(f"No rows found for station {station_name!r} in {path}")

    raw_records.sort(key=lambda record: record.date)
    reconstructed = [_reconstruct_temperatures(record) for record in raw_records]
    return _fill_remaining_temperatures(reconstructed)


def train_test_split(
    records: list[WeatherRecord],
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
) -> tuple[list[WeatherRecord], list[WeatherRecord]]:
    """Split records into inclusive calendar windows."""

    train = [record for record in records if train_start <= record.date <= train_end]
    test = [record for record in records if test_start <= record.date <= test_end]
    if not train:
        raise ValueError("Training split is empty.")
    if not test:
        raise ValueError("Testing split is empty.")
    return train, test


def parse_iso_date(value: str) -> date:
    """Parse an ISO date argument."""

    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_date(value: str) -> date:
    # Support both YYYYMMDD (old CDO export) and YYYY-MM-DD (new CDO export)
    if "-" in value:
        return datetime.strptime(value, "%Y-%m-%d").date()
    return datetime.strptime(value, "%Y%m%d").date()


def _parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    parsed = float(value)
    if parsed == MISSING_VALUE:
        return None
    return parsed


def _clean_precipitation(values: dict[str, float | None]) -> dict[str, float | None]:
    cleaned = dict(values)
    for column in ("PRCP", "SNWD", "SNOW"):
        if cleaned[column] is None:
            cleaned[column] = 0.0
    return cleaned


def _reconstruct_temperatures(record: WeatherRecord) -> WeatherRecord:
    values = dict(record.values)
    tmax = values["TMAX"]
    tmin = values["TMIN"]
    tobs = values["TOBS"]

    if tobs is None and tmax is not None and tmin is not None:
        values["TOBS"] = (tmax + tmin) / 2.0
    if tmax is None and tmin is not None and values["TOBS"] is not None:
        values["TMAX"] = 2.0 * values["TOBS"] - tmin
    if tmin is None and tmax is not None and values["TOBS"] is not None:
        values["TMIN"] = 2.0 * values["TOBS"] - tmax

    return WeatherRecord(record.station, record.station_name, record.date, values)


def _fill_remaining_temperatures(records: list[WeatherRecord]) -> list[WeatherRecord]:
    by_day: dict[tuple[int, int], dict[str, list[float]]] = {}
    global_values: dict[str, list[float]] = {"TMAX": [], "TMIN": [], "TOBS": []}

    for record in records:
        key = (record.date.month, record.date.day)
        by_day.setdefault(key, {"TMAX": [], "TMIN": [], "TOBS": []})
        for column in global_values:
            value = record.values[column]
            if value is not None:
                by_day[key][column].append(value)
                global_values[column].append(value)

    day_means = {
        key: {column: mean(values) for column, values in columns.items() if values}
        for key, columns in by_day.items()
    }
    missing_columns = [column for column, values in global_values.items() if not values]
    if missing_columns:
        columns = ", ".join(missing_columns)
        raise ValueError(f"Cannot fill temperature columns with no observed values: {columns}")

    global_means = {column: mean(values) for column, values in global_values.items()}

    filled: list[WeatherRecord] = []
    for record in records:
        values = dict(record.values)
        key = (record.date.month, record.date.day)
        for column in global_values:
            if values[column] is None:
                values[column] = day_means.get(key, {}).get(column, global_means[column])
        filled.append(WeatherRecord(record.station, record.station_name, record.date, values))
    return filled

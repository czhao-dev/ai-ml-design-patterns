"""Minimal SVG reporting utilities."""

from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path


def write_actual_vs_predicted_svg(
    path: str | Path,
    dates: list[date],
    actual: list[float],
    predicted: list[float],
    title: str,
    ylabel: str,
) -> None:
    """Write a dependency-free line chart as SVG."""

    if not dates:
        raise ValueError("Cannot plot an empty date series.")
    if len(dates) != len(actual) or len(actual) != len(predicted):
        raise ValueError("Dates, actual values, and predictions must have equal length.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 960
    height = 420
    left = 72
    right = 24
    top = 44
    bottom = 54
    plot_width = width - left - right
    plot_height = height - top - bottom

    y_min = min(actual + predicted)
    y_max = max(actual + predicted)
    padding = (y_max - y_min) * 0.08 or 1.0
    y_min -= padding
    y_max += padding

    def x_position(index: int) -> float:
        if len(dates) == 1:
            return left
        return left + plot_width * index / (len(dates) - 1)

    def y_position(value: float) -> float:
        return top + plot_height * (1.0 - (value - y_min) / (y_max - y_min))

    actual_points = " ".join(
        f"{x_position(index):.2f},{y_position(value):.2f}"
        for index, value in enumerate(actual)
    )
    predicted_points = " ".join(
        f"{x_position(index):.2f},{y_position(value):.2f}"
        for index, value in enumerate(predicted)
    )

    y_ticks = _ticks(y_min, y_max, 5)
    x_ticks = _date_ticks(dates, 6)

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="420" viewBox="0 0 960 420">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#202124}",
        ".axis{stroke:#5f6368;stroke-width:1}",
        ".grid{stroke:#e8eaed;stroke-width:1}",
        ".actual{fill:none;stroke:#1967d2;stroke-width:2.2}",
        ".predicted{fill:none;stroke:#d93025;stroke-width:2.2;stroke-dasharray:6 4}",
        "</style>",
        f'<text x="{left}" y="26" font-size="18" font-weight="700">{escape(title)}</text>',
        f'<text x="20" y="{top + plot_height / 2:.0f}" font-size="12" transform="rotate(-90 20,{top + plot_height / 2:.0f})">{escape(ylabel)}</text>',
        f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
    ]

    for tick in y_ticks:
        y = y_position(tick)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{tick:.2f}</text>')

    for index, label in x_ticks:
        x = x_position(index)
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>')
        lines.append(f'<text x="{x:.2f}" y="{height - 20}" text-anchor="middle" font-size="11">{escape(label)}</text>')

    lines.extend(
        [
            f'<polyline class="actual" points="{actual_points}"/>',
            f'<polyline class="predicted" points="{predicted_points}"/>',
            '<line x1="735" y1="24" x2="775" y2="24" class="actual"/>',
            '<text x="782" y="28" font-size="12">Actual</text>',
            '<line x1="835" y1="24" x2="875" y2="24" class="predicted"/>',
            '<text x="882" y="28" font-size="12">Predicted</text>',
            "</svg>",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _ticks(minimum: float, maximum: float, count: int) -> list[float]:
    if count <= 1:
        return [minimum]
    step = (maximum - minimum) / (count - 1)
    return [minimum + step * index for index in range(count)]


def _date_ticks(dates: list[date], count: int) -> list[tuple[int, str]]:
    if len(dates) <= count:
        return [(index, day.isoformat()) for index, day in enumerate(dates)]
    step = (len(dates) - 1) / (count - 1)
    ticks = []
    for tick in range(count):
        index = round(tick * step)
        ticks.append((index, dates[index].strftime("%b %d")))
    return ticks

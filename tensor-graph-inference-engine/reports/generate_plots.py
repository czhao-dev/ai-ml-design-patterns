"""Generate benchmark comparison charts from persisted results.

Reads benchmark_results.csv -- no re-running the benchmark, purely a
visualization of already-measured numbers from scripts/03_benchmark.py.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORTS_DIR = Path(__file__).resolve().parent

THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink_secondary="#52514e",
        muted="#898781", grid="#e1e0d9", baseline="#c3c2b7",
        colors=["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"],
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", ink_secondary="#c3c2b7",
        muted="#898781", grid="#2c2c2a", baseline="#383835",
        colors=["#3987e5", "#199e70", "#c98500", "#008300", "#9085e9", "#e66767"],
    ),
}

VARIANT_LABELS = {"int8": "INT8 engine", "fp32": "fp32 reference"}


def load_csv(name="benchmark_results.csv"):
    with open(REPORTS_DIR / name) as f:
        return list(csv.DictReader(f))


def style_axes(ax, t):
    ax.set_facecolor(t["surface"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(t["muted"])
    ax.tick_params(colors=t["ink_secondary"], labelsize=9)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["ink_secondary"])
    ax.yaxis.label.set_color(t["ink_secondary"])


def add_title(fig, title, subtitle, t):
    fig.suptitle(title, x=0.02, y=0.985, ha="left", va="top", fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.02, 0.905, subtitle, ha="left", va="top", fontsize=8.5, color=t["muted"])


def _series(rows, variant, field):
    pts = sorted(
        ((int(r["batch"]), float(r[field])) for r in rows if r["variant"] == variant),
        key=lambda p: p[0],
    )
    return [p[0] for p in pts], [p[1] for p in pts]


def _plot_two_series(rows, theme_name, field, ylabel, title, out_name, fmt_subtitle):
    t = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=200)
    fig.patch.set_facecolor(t["surface"])

    batches, int8_vals = _series(rows, "int8", field)
    _, fp32_vals = _series(rows, "fp32", field)

    ax.plot(batches, int8_vals, color=t["colors"][0], linewidth=2.2, linestyle="-",
            marker="o", markersize=6, label=VARIANT_LABELS["int8"], zorder=4)
    ax.plot(batches, fp32_vals, color=t["colors"][2], linewidth=1.8, linestyle="--",
            marker="s", markersize=5.5, label=VARIANT_LABELS["fp32"], zorder=3)

    ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")

    largest_batch = max(batches)
    ratio = int8_vals[batches.index(largest_batch)] / fp32_vals[batches.index(largest_batch)]
    subtitle = fmt_subtitle(ratio, largest_batch)

    add_title(fig, title, subtitle, t)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2,
        frameon=False, fontsize=8.8, labelcolor=t["ink_secondary"],
    )
    style_axes(ax, t)
    fig.tight_layout(rect=[0, 0.08, 1, 0.88])
    out = REPORTS_DIR / f"{out_name}_{theme_name}.png"
    fig.savefig(out, dpi=200, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)
    return out


def plot_throughput_vs_batch(rows, theme_name):
    def subtitle(ratio, batch):
        if ratio > 1:
            return f"INT8 engine has {ratio:.1f}x higher throughput than the fp32 baseline at batch {batch}."
        return f"INT8 engine has {1 / ratio:.1f}x lower throughput than the fp32 baseline at batch {batch}."

    return _plot_two_series(
        rows, theme_name, "throughput_rows_per_sec", "Throughput (rows/sec, log scale)",
        "Inference throughput vs. batch size", "throughput_vs_batch", subtitle,
    )


def plot_latency_vs_batch(rows, theme_name):
    def subtitle(ratio, batch):
        if ratio > 1:
            return f"INT8 engine has {ratio:.1f}x higher mean latency than the fp32 baseline at batch {batch}."
        return f"INT8 engine has {1 / ratio:.1f}x lower mean latency than the fp32 baseline at batch {batch}."

    return _plot_two_series(
        rows, theme_name, "mean_latency_ms", "Mean latency per call (ms, log scale)",
        "Per-call latency vs. batch size", "latency_vs_batch", subtitle,
    )


def main():
    rows = load_csv()
    for theme_name in ("light", "dark"):
        plot_throughput_vs_batch(rows, theme_name)
        plot_latency_vs_batch(rows, theme_name)
    print("Wrote plots to", REPORTS_DIR)


if __name__ == "__main__":
    main()

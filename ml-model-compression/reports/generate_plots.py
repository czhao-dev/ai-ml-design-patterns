"""Generate compression-tradeoff plots from reports/results.json.

Reads only persisted benchmark results (no retraining, no API calls) and
renders light/dark themed PNGs into reports/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "results.json")) as f:
    rows = json.load(f)

TECHNIQUE_COLOR_IDX = {
    "Baseline": 3,     # green
    "Pruning": 0,      # blue
    "Quantization": 2, # amber
    "Distillation": 4, # purple
}


def style_axes(ax, t):
    ax.set_facecolor(t["surface"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(t["muted"])
    ax.spines["bottom"].set_color(t["muted"])
    ax.tick_params(colors=t["ink_secondary"])
    ax.grid(True, color=t["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["ink_secondary"])
    ax.yaxis.label.set_color(t["ink_secondary"])
    ax.title.set_color(t["ink"])


def plot_size_vs_accuracy(theme_name):
    t = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=200)
    fig.patch.set_facecolor(t["surface"])

    seen_techniques = []
    for row in rows:
        tech = row["technique"]
        if tech not in seen_techniques:
            seen_techniques.append(tech)

    for tech in seen_techniques:
        xs = [r["size_mb"] for r in rows if r["technique"] == tech]
        ys = [r["accuracy"] * 100 for r in rows if r["technique"] == tech]
        color = t["colors"][TECHNIQUE_COLOR_IDX.get(tech, 5)]
        ax.scatter(xs, ys, s=90, color=color, label=tech, zorder=3,
                   edgecolor=t["surface"], linewidth=1.2)

    ax.set_xscale("log")
    ax.set_xlabel("Model size (MB, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Compression Tradeoff — Model Size vs. Accuracy", fontsize=13, fontweight="bold", loc="left", color=t["ink"])
    ax.set_ylim(45, 103)
    style_axes(ax, t)

    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), labelcolor=t["ink_secondary"])

    fig.text(0.01, -0.02,
              "Each point is one benchmarked variant (14 total) on the fixed 1,200-image held-out split; "
              "size includes serialized weights only, CPU-measured.",
              fontsize=8, color=t["muted"], ha="left")

    fig.tight_layout()
    out = os.path.join(HERE, f"size_vs_accuracy_{theme_name}.png")
    fig.savefig(out, dpi=200, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_pruning_accuracy_vs_sparsity(theme_name):
    t = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=200)
    fig.patch.set_facecolor(t["surface"])

    pruning_rows = [r for r in rows if r["technique"] == "Pruning"]
    baseline_acc = next(r["accuracy"] for r in rows if r["variant"] == "PyTorch CNN (FP32 baseline)") * 100

    for prune_type, color_idx, marker in [("unstructured", 0, "o"), ("structured", 5, "s")]:
        subset = sorted(
            (r for r in pruning_rows if r["prune_type"] == prune_type),
            key=lambda r: r["sparsity"],
        )
        xs = [0.0] + [r["sparsity"] * 100 for r in subset]
        ys = [baseline_acc] + [r["accuracy"] * 100 for r in subset]
        ax.plot(xs, ys, marker=marker, color=t["colors"][color_idx], linewidth=2,
                markersize=7, label=prune_type.capitalize(), zorder=3)

    ax.axhline(baseline_acc, color=t["baseline"], linewidth=1, linestyle="--", zorder=2)
    ax.annotate("FP32 baseline", xy=(0, baseline_acc), xytext=(8, baseline_acc - 10),
                fontsize=8, color=t["muted"])

    ax.set_xlabel("Sparsity (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Pruning — Accuracy vs. Sparsity", fontsize=13, fontweight="bold", loc="left", color=t["ink"])
    ax.set_ylim(0, 105)
    style_axes(ax, t)

    ax.legend(frameon=False, loc="lower left", labelcolor=t["ink_secondary"])

    fig.text(0.01, -0.02,
              "No fine-tuning after pruning. Structured pruning collapses to the class prior (~52%) by 40% sparsity; "
              "unstructured holds up to 60% before collapsing at 80%.",
              fontsize=8, color=t["muted"], ha="left")

    fig.tight_layout()
    out = os.path.join(HERE, f"pruning_accuracy_vs_sparsity_{theme_name}.png")
    fig.savefig(out, dpi=200, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


for theme_name in THEMES:
    plot_size_vs_accuracy(theme_name)
    plot_pruning_accuracy_vs_sparsity(theme_name)

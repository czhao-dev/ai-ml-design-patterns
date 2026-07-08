"""Generate model-comparison charts from the numeric table in reports/results_summary.md.

This project has no persisted CSV/JSON metrics files (checked: no files under
reports/, data/, or logs beyond the raw image archive) -- the "Model Results"
table in results_summary.md / README.md is the only structured numeric data
that exists. Values below are transcribed verbatim from that table; no numbers
are invented and nothing is retrained.
"""
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

# Transcribed from reports/results_summary.md ("Model Results" table),
# held-out validation split (1,200 images, 20% of images_dataSAT).
MODELS = ["Keras CNN", "PyTorch CNN", "Keras CNN-ViT\nHybrid", "PyTorch CNN-ViT\nHybrid"]
METRICS = {
    "Accuracy": [0.9933, 0.9983, 0.9942, 0.9967],
    "Precision": [1.0000, 0.9965, 0.9966, 0.9983],
    "Recall": [0.9867, 1.0000, 0.9917, 0.9950],
    "F1 Score": [0.9933, 0.9983, 0.9942, 0.9967],
}
LOSS = [0.0257, 0.0041, 0.1138, 0.0104]


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


def plot_metric_comparison(theme_name):
    t = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
    fig.patch.set_facecolor(t["surface"])

    metric_names = list(METRICS.keys())
    n_models = len(MODELS)
    n_metrics = len(metric_names)
    bar_width = 0.8 / n_metrics
    x = np.arange(n_models)

    for i, metric in enumerate(metric_names):
        vals = METRICS[metric]
        offset = (i - (n_metrics - 1) / 2) * bar_width
        ax.bar(
            x + offset, vals, width=bar_width * 0.92,
            color=t["colors"][i], label=metric, zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Score (held-out validation split)")
    ax.set_ylim(0.97, 1.005)
    add_title(
        fig,
        "Model comparison: accuracy, precision, recall, F1",
        "6,000-tile balanced satellite dataset, 1,200-image held-out validation split -- y-axis zoomed to show separation.",
        t,
    )
    legend = ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=4,
        frameon=False, fontsize=9, labelcolor=t["ink_secondary"],
    )
    style_axes(ax, t)
    fig.tight_layout(rect=[0, 0.06, 1, 0.86])
    out = REPORTS_DIR / f"model_metric_comparison_{theme_name}.png"
    fig.savefig(out, dpi=200, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)
    return out


def plot_loss_comparison(theme_name):
    t = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=200)
    fig.patch.set_facecolor(t["surface"])

    x = np.arange(len(MODELS))
    bars = ax.bar(x, LOSS, width=0.55, color=t["colors"][:len(MODELS)], zorder=3)
    for b, v in zip(bars, LOSS):
        ax.annotate(
            f"{v:.4f}", (b.get_x() + b.get_width() / 2, v),
            xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom", fontsize=9.5, color=t["ink_secondary"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Validation loss (lower is better)")
    ax.set_ylim(0, max(LOSS) * 1.3)
    add_title(
        fig,
        "Validation loss by model",
        "PyTorch CNN has the lowest loss; the Keras CNN-ViT hybrid the highest, despite a comparable F1 score.",
        t,
    )
    style_axes(ax, t)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    out = REPORTS_DIR / f"model_loss_comparison_{theme_name}.png"
    fig.savefig(out, dpi=200, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    for theme_name in ("light", "dark"):
        plot_metric_comparison(theme_name)
        plot_loss_comparison(theme_name)
    print("Wrote plots to", REPORTS_DIR)


if __name__ == "__main__":
    main()

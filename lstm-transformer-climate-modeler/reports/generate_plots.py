"""Generate LSTM vs. Transformer comparison plots from reports/metrics_tf.json.

Reads only persisted evaluation results (no retraining, no API calls) and
renders light/dark themed PNGs into reports/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

with open(os.path.join(HERE, "metrics_tf.json")) as f:
    metrics_tf = json.load(f)

targets = metrics_tf["targets"]  # ["PRCP", "SNOW", "TOBS"]
target_labels = {"PRCP": "Precipitation", "SNOW": "Snowfall", "TOBS": "Temperature"}
lstm_metrics = metrics_tf["models"]["lstm"]["metrics"]
tf_metrics = metrics_tf["models"]["transformer"]["metrics"]


def style_axes(ax, t):
    ax.set_facecolor(t["surface"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(t["muted"])
    ax.spines["bottom"].set_color(t["muted"])
    ax.tick_params(colors=t["ink_secondary"])
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["ink_secondary"])
    ax.yaxis.label.set_color(t["ink_secondary"])
    ax.title.set_color(t["ink"])


def plot_rmse_comparison(theme_name):
    t = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=200)
    fig.patch.set_facecolor(t["surface"])

    x = np.arange(len(targets))
    width = 0.32

    lstm_rmse = [lstm_metrics[k]["rmse"] for k in targets]
    tf_rmse = [tf_metrics[k]["rmse"] for k in targets]

    b1 = ax.bar(x - width / 2, lstm_rmse, width, label="LSTM", color=t["colors"][0],
                zorder=3, edgecolor=t["surface"], linewidth=2)
    b2 = ax.bar(x + width / 2, tf_rmse, width, label="Transformer", color=t["colors"][1],
                zorder=3, edgecolor=t["surface"], linewidth=2)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color=t["ink_secondary"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{target_labels[k]}\n({k})" for k in targets], color=t["ink_secondary"])
    ax.set_ylabel("RMSE (7-day-ahead, test set)")
    ax.set_title("LSTM vs. Transformer — Forecast RMSE by Target", fontsize=13, fontweight="bold", loc="left", color=t["ink"])
    style_axes(ax, t)

    legend = ax.legend(frameon=False, loc="upper left", labelcolor=t["ink_secondary"])

    fig.text(0.01, -0.02,
              "Averaged over all 7 forecast horizons · 717 test windows · trained on 1960-2017, tested on 2018-2019",
              fontsize=8, color=t["muted"], ha="left")

    fig.tight_layout()
    out = os.path.join(HERE, f"lstm_vs_transformer_rmse_{theme_name}.png")
    fig.savefig(out, dpi=200, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_r2_comparison(theme_name):
    t = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=200)
    fig.patch.set_facecolor(t["surface"])

    x = np.arange(len(targets))
    width = 0.32

    lstm_r2 = [lstm_metrics[k]["r2"] for k in targets]
    tf_r2 = [tf_metrics[k]["r2"] for k in targets]

    ax.axhline(0, color=t["baseline"], linewidth=1, zorder=2)

    b1 = ax.bar(x - width / 2, lstm_r2, width, label="LSTM", color=t["colors"][0],
                zorder=3, edgecolor=t["surface"], linewidth=2)
    b2 = ax.bar(x + width / 2, tf_r2, width, label="Transformer", color=t["colors"][1],
                zorder=3, edgecolor=t["surface"], linewidth=2)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            offset = 3 if h >= 0 else -3
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, offset), textcoords="offset points",
                        ha="center", va=va, fontsize=8.5, color=t["ink_secondary"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{target_labels[k]}\n({k})" for k in targets], color=t["ink_secondary"])
    ax.set_ylabel("R² (7-day-ahead, test set)")
    ax.set_title("LSTM vs. Transformer — Forecast R² by Target", fontsize=13, fontweight="bold", loc="left", color=t["ink"])
    style_axes(ax, t)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8, zorder=0)

    ax.legend(frameon=False, loc="upper left", labelcolor=t["ink_secondary"])

    fig.text(0.01, -0.02,
              "R² near 0 for PRCP/SNOW reflects sparse, event-driven precipitation; TOBS is where both models fit well.",
              fontsize=8, color=t["muted"], ha="left")

    fig.tight_layout()
    out = os.path.join(HERE, f"lstm_vs_transformer_r2_{theme_name}.png")
    fig.savefig(out, dpi=200, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


for theme_name in THEMES:
    plot_rmse_comparison(theme_name)
    plot_r2_comparison(theme_name)

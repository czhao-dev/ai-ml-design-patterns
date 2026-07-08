"""Generate result-comparison plots from the numbers already reported in
reports/results_summary.md and README.md (no retraining, no new experiments).

Data sources (persisted, existing results):
  - IMDb full-scale (129,720 labeled movies) holdout RMSE/MAE:
    reports/results_summary.md -> "GNN HOLDOUT ... RMSE=1.3490, MAE=1.1358"
    and "Global Mean baseline ... RMSE=1.1251, MAE=0.8772"
  - MovieLens (ml-latest-small) Test RMSE/MAE table: README.md "Results" ->
    MovieLens track table (GNN / Global Mean / User Mean / Item Mean).

Run with the project's plotting venv, e.g.:
  /path/to/plotenv/bin/python reports/generate_plots.py
"""

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


def style_axes(ax, t):
    ax.set_facecolor(t["surface"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(t["baseline"])
    ax.spines["bottom"].set_color(t["baseline"])
    ax.tick_params(colors=t["ink_secondary"], labelsize=10)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["ink_secondary"])
    ax.yaxis.label.set_color(t["ink_secondary"])
    # NOTE: axes.titlecolor defaults to "auto" in this matplotlib install,
    # which re-applies text.color at draw time and silently overrides
    # ax.title.set_color() called after the fact. Color must instead be
    # passed directly to ax.set_title(..., color=...) at creation time.


def grouped_bar_chart(
    methods, series_labels, series_values, ylabel, title, subtitle, outpath_base
):
    """series_values: list (one per series) of lists (one per method)."""
    n_methods = len(methods)
    n_series = len(series_labels)
    x = np.arange(n_methods)
    width = 0.7 / n_series

    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=t["surface"])
        ax.set_facecolor(t["surface"])

        for i, (label, values) in enumerate(zip(series_labels, series_values)):
            offset = (i - (n_series - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                values,
                width=width * 0.9,
                label=label,
                color=t["colors"][i % len(t["colors"])],
                zorder=3,
            )
            for rect, v in zip(bars, values):
                ax.annotate(
                    f"{v:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=t["ink_secondary"],
                )

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(
            title, fontsize=13, fontweight="bold", loc="left", pad=28,
            color=t["ink"],
        )
        fig.text(
            0.125, 0.90, subtitle, fontsize=9.5, color=t["muted"], ha="left",
        )
        style_axes(ax, t)
        ax.legend(
            frameon=False,
            labelcolor=t["ink_secondary"],
            fontsize=9.5,
            loc="upper right",
        )
        ymax = max(max(v) for v in series_values)
        ax.set_ylim(0, ymax * 1.22)

        fig.tight_layout(rect=(0, 0, 1, 0.92))
        outpath = f"{outpath_base}_{mode}.png"
        fig.savefig(outpath, dpi=200, facecolor=t["surface"])
        plt.close(fig)
        print(f"wrote {outpath}")


if __name__ == "__main__":
    # --- Chart 1: IMDb full-scale (129,720 labeled movies) holdout test set ---
    grouped_bar_chart(
        methods=["GNN (heterogeneous,\nholdout)", "Global Mean\n(constant baseline)"],
        series_labels=["RMSE", "MAE"],
        series_values=[[1.3490, 1.1251], [1.1358, 0.8772]],
        ylabel="Rating error (lower is better)",
        title="IMDb full-scale: the GNN does not beat a constant-mean baseline",
        subtitle="Held-out test set, 19,458 of 129,720 labeled movies never seen during training",
        outpath_base="imdb_full_holdout_error",
    )

    # --- Chart 2: MovieLens ml-latest-small test RMSE/MAE by method ---
    grouped_bar_chart(
        methods=["GNN\n(heterogeneous)", "Global Mean", "User Mean", "Item Mean"],
        series_labels=["RMSE", "MAE"],
        series_values=[[0.99, 1.12, 1.03, 1.05], [0.78, 0.92, 0.80, 0.83]],
        ylabel="Rating error (lower is better)",
        title="MovieLens: the GNN beats every non-personalized baseline",
        subtitle="ml-latest-small, 610 users / 9,742 movies / 100K ratings, test split",
        outpath_base="movielens_rmse_mae",
    )

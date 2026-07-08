"""Generate comparison charts from persisted uplift-modeling results.

Reads uplift_model_comparison.csv and revenue_policy_comparison.csv -- no
retraining, no new experiments, purely a visualization of already-computed
results from scripts/07_evaluate_uplift.py and scripts/08_revenue_simulation.py.
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

MODEL_ORDER = [
    "Baseline response model (ignores treatment)",
    "T-learner",
    "X-learner",
    "Causal forest",
]
MODEL_LABELS = ["Baseline\n(ignores treatment)", "T-learner", "X-learner", "Causal forest"]
POLICY_ORDER = [
    "Baseline response model (ignores treatment)",
    "T-learner",
    "X-learner",
    "Causal forest",
    "Random targeting",
]
POLICY_LABELS = ["Baseline", "T-learner", "X-learner", "Causal forest", "Random targeting"]


def load_csv(name):
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


def plot_qini_comparison(rows, theme_name):
    t = THEMES[theme_name]
    by_model = {r["model"]: float(r["qini_coefficient"]) for r in rows}
    vals = [by_model[m] for m in MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=200)
    fig.patch.set_facecolor(t["surface"])

    x = np.arange(len(MODEL_ORDER))
    colors = [t["colors"][3] if v > 0 else t["colors"][5] for v in vals]
    bars = ax.bar(x, vals, width=0.55, color=colors, zorder=3)

    for b, v in zip(bars, vals):
        va = "bottom" if v >= 0 else "top"
        offset = 12 if v >= 0 else -12
        ax.annotate(
            f"{v:+.0f}", (b.get_x() + b.get_width() / 2, v),
            xytext=(0, offset), textcoords="offset points",
            ha="center", va=va, fontsize=9.5, color=t["ink_secondary"],
        )

    ax.axhline(0, color=t["baseline"], linewidth=1.2, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_ylabel("Qini coefficient (held-out test set)")
    ymin, ymax = min(vals), max(vals)
    pad = (ymax - ymin) * 0.25
    ax.set_ylim(ymin - pad, ymax + pad)
    add_title(
        fig,
        "Uplift model comparison: Qini coefficient",
        "Positive = beats random targeting on held-out RCT data. Only the causal forest is positive.",
        t,
    )
    style_axes(ax, t)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = REPORTS_DIR / f"qini_comparison_{theme_name}.png"
    fig.savefig(out, dpi=200, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)
    return out


def plot_revenue_by_policy(rows, theme_name):
    t = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=200)
    fig.patch.set_facecolor(t["surface"])

    for i, policy in enumerate(POLICY_ORDER):
        pts = sorted(
            ((float(r["targeting_fraction"]), float(r["net_revenue"])) for r in rows if r["policy"] == policy),
            key=lambda p: p[0],
        )
        xs = [p[0] * 100 for p in pts]
        ys = [p[1] for p in pts]
        is_random = policy == "Random targeting"
        ax.plot(
            xs, ys,
            color=t["colors"][i], linewidth=2.2 if not is_random else 1.6,
            linestyle="--" if is_random else "-",
            marker="o", markersize=5,
            label=POLICY_LABELS[i], zorder=4 if not is_random else 3,
        )

    ax.axhline(0, color=t["baseline"], linewidth=1, zorder=1)
    ax.set_xlabel("Targeting fraction (% of customers targeted)")
    ax.set_ylabel("Net incremental revenue ($)")
    add_title(
        fig,
        "Net revenue by targeting policy, across targeting fractions",
        "$2.00 assumed cost per offer -- causal forest loses least at low targeting fractions and is the only policy near breakeven at 5%.",
        t,
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3,
        frameon=False, fontsize=8.8, labelcolor=t["ink_secondary"],
    )
    style_axes(ax, t)
    fig.tight_layout(rect=[0, 0.14, 1, 0.86])
    out = REPORTS_DIR / f"revenue_by_policy_{theme_name}.png"
    fig.savefig(out, dpi=200, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    uplift_rows = load_csv("uplift_model_comparison.csv")
    revenue_rows = load_csv("revenue_policy_comparison.csv")
    for theme_name in ("light", "dark"):
        plot_qini_comparison(uplift_rows, theme_name)
        plot_revenue_by_policy(revenue_rows, theme_name)
    print("Wrote plots to", REPORTS_DIR)


if __name__ == "__main__":
    main()

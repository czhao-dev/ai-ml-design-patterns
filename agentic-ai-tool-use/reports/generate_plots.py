"""Generate comparison charts from persisted benchmark metrics (reports/*_metrics.json).

Reads react_metrics.json, plan_execute_metrics.json, reflexion_metrics.json --
no retraining, no new API calls, purely a visualization of already-run results.
"""
import json
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

ARCH_FILES = {
    "ReAct": "react_metrics.json",
    "Plan-and-Execute": "plan_execute_metrics.json",
    "Reflexion": "reflexion_metrics.json",
}
CATEGORIES = ["arithmetic", "code_exec", "error_recovery", "multihop_qa"]
CATEGORY_LABELS = ["Arithmetic", "Code exec", "Error recovery", "Multi-hop QA"]


def load_metrics():
    data = {}
    for arch, fname in ARCH_FILES.items():
        with open(REPORTS_DIR / fname) as f:
            data[arch] = json.load(f)
    return data


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


def plot_success_by_category(data, theme_name):
    t = THEMES[theme_name]
    archs = list(ARCH_FILES.keys())
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    fig.patch.set_facecolor(t["surface"])

    n_arch = len(archs)
    n_cat = len(CATEGORIES)
    bar_width = 0.8 / n_arch
    x = np.arange(n_cat)

    for i, arch in enumerate(archs):
        vals = [data[arch]["success_rate_by_category"][c] * 100 for c in CATEGORIES]
        offset = (i - (n_arch - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, vals, width=bar_width * 0.92,
            color=t["colors"][i], label=arch, zorder=3,
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=7.5, color=t["ink_secondary"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_LABELS)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 112)
    add_title(
        fig,
        "Success rate by task category, per agent architecture",
        "gpt-4.1, 35-task benchmark (9 arithmetic, 9 code exec, 8 error recovery, 9 multi-hop QA)",
        t,
    )
    legend = ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3,
        frameon=False, fontsize=9, labelcolor=t["ink_secondary"],
    )
    style_axes(ax, t)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out = REPORTS_DIR / f"success_by_category_{theme_name}.png"
    fig.savefig(out, dpi=200, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cost_vs_success(data, theme_name):
    t = THEMES[theme_name]
    archs = list(ARCH_FILES.keys())
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    fig.patch.set_facecolor(t["surface"])

    costs = [data[a]["estimated_cost_usd"] for a in archs]
    success = [data[a]["overall_success_rate"] * 100 for a in archs]
    x = np.arange(len(archs))
    bar_width = 0.55

    bars = ax.bar(x, costs, width=bar_width, color=t["colors"][:len(archs)], zorder=3)
    for b, c, s in zip(bars, costs, success):
        ax.text(
            b.get_x() + b.get_width() / 2, c + max(costs) * 0.02,
            f"${c:.3f}\n({s:.1f}% success)",
            ha="center", va="bottom", fontsize=8.5, color=t["ink_secondary"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(archs)
    ax.set_ylabel("Estimated cost (USD, full 35-task run)")
    ax.set_ylim(0, max(costs) * 1.35)
    add_title(
        fig,
        "Benchmark cost by architecture",
        "Overall success rate annotated per bar -- gpt-4.1, same task set and tools",
        t,
    )
    style_axes(ax, t)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = REPORTS_DIR / f"cost_comparison_{theme_name}.png"
    fig.savefig(out, dpi=200, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    data = load_metrics()
    for theme_name in ("light", "dark"):
        plot_success_by_category(data, theme_name)
        plot_cost_vs_success(data, theme_name)
    print("Wrote plots to", REPORTS_DIR)


if __name__ == "__main__":
    main()

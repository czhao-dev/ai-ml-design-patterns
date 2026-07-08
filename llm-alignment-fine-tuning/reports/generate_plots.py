"""Generate alignment-pipeline comparison plots from the persisted metrics
JSON files already written by scripts 01-04 (no retraining, no new runs).

Data sources (persisted, existing results):
  - reports/ppo_metrics.json   -> policies.POSITIVE/NEGATIVE.mean_reward_before/after
  - reports/dpo_metrics.json   -> mean_proxy_reward_base/dpo, eval_reward_accuracy
  - reports/reward_metrics.json -> pairwise_ranking_accuracy

PPO's mean_reward_before/after and DPO's mean_proxy_reward_base/dpo are on
the same scale on purpose: DPO's proxy reward is "reused from the PPO script,
per project scope" (see reports/results_summary.md), so this is a genuine
apples-to-apples before/after comparison, not a chart-time normalization.

Run with the project's plotting venv, e.g.:
  /path/to/plotenv/bin/python reports/generate_plots.py
"""

import json
from pathlib import Path

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

REPORTS_DIR = Path(__file__).parent


def load_json(name):
    with open(REPORTS_DIR / name) as f:
        return json.load(f)


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
    # which re-applies text.color at draw time and silently overrides any
    # ax.title.set_color() called after the fact. Title color must instead
    # be passed directly to ax.set_title(..., color=...) at creation time.


def reward_before_after_chart(policies, values_before, values_after, outpath_base):
    x = np.arange(len(policies))
    width = 0.32

    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=t["surface"])
        ax.set_facecolor(t["surface"])

        bars_before = ax.bar(
            x - width / 2, values_before, width=width * 0.9,
            label="Before", color=t["colors"][0], zorder=3,
        )
        bars_after = ax.bar(
            x + width / 2, values_after, width=width * 0.9,
            label="After", color=t["colors"][1], zorder=3,
        )
        for bars in (bars_before, bars_after):
            for rect in bars:
                h = rect.get_height()
                va = "bottom" if h >= 0 else "top"
                offset = 3 if h >= 0 else -3
                ax.annotate(
                    f"{h:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center", va=va, fontsize=9,
                    color=t["ink_secondary"],
                )

        ax.axhline(0, color=t["baseline"], linewidth=1, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(policies, fontsize=10)
        ax.set_ylabel("Sentiment-proxy reward")
        ax.set_title(
            "Sentiment-proxy reward moves in the intended direction\n"
            "for every RL/preference-optimization policy",
            fontsize=13, fontweight="bold", loc="left", pad=34, color=t["ink"],
        )
        fig.text(
            0.125, 0.92,
            "PPO (positive/negative sentiment) and DPO, before vs. after training "
            "— same reward scale",
            fontsize=9.5, color=t["muted"], ha="left",
        )
        style_axes(ax, t)
        ax.legend(
            frameon=False, labelcolor=t["ink_secondary"], fontsize=9.5,
            loc="upper left",
        )
        ymin = min(min(values_before), min(values_after))
        ymax = max(max(values_before), max(values_after))
        pad = (ymax - ymin) * 0.2
        ax.set_ylim(ymin - pad, ymax + pad)

        fig.tight_layout(rect=(0, 0, 1, 0.90))
        outpath = f"{outpath_base}_{mode}.png"
        fig.savefig(outpath, dpi=200, facecolor=t["surface"])
        plt.close(fig)
        print(f"wrote {outpath}")


def accuracy_bar_chart(labels, values, outpath_base):
    x = np.arange(len(labels))

    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(6.5, 5), facecolor=t["surface"])
        ax.set_facecolor(t["surface"])

        bars = ax.bar(
            x, values, width=0.5,
            color=[t["colors"][0], t["colors"][2]][: len(labels)],
            zorder=3,
        )
        for rect, v in zip(bars, values):
            ax.annotate(
                f"{v:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=10,
                color=t["ink_secondary"],
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Held-out pairwise accuracy")
        ax.set_title(
            "Discriminating preferences is easier than\ngenerating better text",
            fontsize=13, fontweight="bold", loc="left", pad=34, color=t["ink"],
        )
        fig.text(
            0.14, 0.92,
            "Reward model vs. DPO's implicit reward, on held-out chosen/rejected pairs",
            fontsize=9.5, color=t["muted"], ha="left",
        )
        style_axes(ax, t)
        ax.set_ylim(0, 1.15)
        ax.axhline(0.5, color=t["baseline"], linewidth=1, linestyle="--", zorder=2)
        ax.annotate(
            "chance", xy=(len(labels) - 0.5, 0.5), xytext=(4, 2),
            textcoords="offset points", fontsize=8.5, color=t["muted"],
        )

        fig.tight_layout(rect=(0, 0, 1, 0.90))
        outpath = f"{outpath_base}_{mode}.png"
        fig.savefig(outpath, dpi=200, facecolor=t["surface"])
        plt.close(fig)
        print(f"wrote {outpath}")


if __name__ == "__main__":
    ppo = load_json("ppo_metrics.json")
    dpo = load_json("dpo_metrics.json")
    reward = load_json("reward_metrics.json")

    pos = ppo["policies"]["POSITIVE"]
    neg = ppo["policies"]["NEGATIVE"]

    # --- Chart 1: reward before/after by policy (PPO x2, DPO) ---
    reward_before_after_chart(
        policies=["PPO\npositive-sentiment", "PPO\nnegative-sentiment", "DPO"],
        values_before=[
            pos["mean_reward_before"],
            neg["mean_reward_before"],
            dpo["mean_proxy_reward_base"],
        ],
        values_after=[
            pos["mean_reward_after"],
            neg["mean_reward_after"],
            dpo["mean_proxy_reward_dpo"],
        ],
        outpath_base=str(REPORTS_DIR / "reward_before_after_by_policy"),
    )

    # --- Chart 2: discriminative accuracy, reward model vs. DPO ---
    accuracy_bar_chart(
        labels=["Reward Model\n(pairwise ranking)", "DPO\n(eval reward accuracy)"],
        values=[reward["pairwise_ranking_accuracy"], dpo["eval_reward_accuracy"]],
        outpath_base=str(REPORTS_DIR / "preference_accuracy_by_technique"),
    )

"""Generate training-curve plots from persisted train_log.jsonl files.

Reads only existing experiment logs under experiments/runs/*/logs/train_log.jsonl
(no retraining, no API calls) and renders light/dark themed PNGs into reports/.
"""
import json
import math
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
RUNS_DIR = os.path.join(HERE, "..", "experiments", "runs")

RUN_LABELS = {"small": "Small (~15M params)", "medium": "Medium (~30M params)"}
RUN_ORDER = ["small", "medium"]


def load_run(run_name):
    path = os.path.join(RUNS_DIR, run_name, "logs", "train_log.jsonl")
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    # De-duplicate by step (a run can be resumed, re-emitting step 0), keep last, sort by step.
    by_step = {}
    for r in records:
        by_step[r["step"]] = r
    return [by_step[s] for s in sorted(by_step)]


runs = {name: load_run(name) for name in RUN_ORDER}


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


def plot_loss_curves(theme_name):
    t = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=200)
    fig.patch.set_facecolor(t["surface"])

    for i, name in enumerate(RUN_ORDER):
        records = runs[name]
        steps = [r["step"] for r in records]
        train_loss = [r["train_loss"] for r in records]
        val_loss = [r["val_loss"] for r in records]
        color = t["colors"][i]
        ax.plot(steps, train_loss, color=color, linewidth=1.6, linestyle="--",
                alpha=0.75, zorder=3, label=f"{RUN_LABELS[name]} — train")
        ax.plot(steps, val_loss, color=color, linewidth=2.2, zorder=4,
                label=f"{RUN_LABELS[name]} — val")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training / Validation Loss — Small vs. Medium", fontsize=13, fontweight="bold", loc="left", color=t["ink"])
    style_axes(ax, t)

    ax.legend(frameon=False, loc="upper right", labelcolor=t["ink_secondary"], fontsize=9)

    fig.text(0.01, -0.02,
              "TinyStories subset, 20,000 steps, identical tokenizer/optimizer schedule for both runs "
              "(dashed = train, solid = validation).",
              fontsize=8, color=t["muted"], ha="left")

    fig.tight_layout()
    out = os.path.join(HERE, f"loss_curves_{theme_name}.png")
    fig.savefig(out, dpi=200, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_perplexity_curves(theme_name):
    t = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=200)
    fig.patch.set_facecolor(t["surface"])

    for i, name in enumerate(RUN_ORDER):
        records = runs[name]
        steps = [r["step"] for r in records]
        ppl = [math.exp(r["val_loss"]) for r in records]
        color = t["colors"][i]
        ax.plot(steps, ppl, color=color, linewidth=2.2, marker="o", markersize=3.5,
                zorder=3, label=RUN_LABELS[name])
        final_step, final_ppl = steps[-1], ppl[-1]
        # Stagger the two end labels vertically so close final values don't overlap.
        y_offset = 9 if i == 0 else -9
        va = "bottom" if i == 0 else "top"
        ax.annotate(f"{final_ppl:.2f}", xy=(final_step, final_ppl),
                    xytext=(8, y_offset), textcoords="offset points",
                    fontsize=9, color=color, va=va, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xlim(right=steps[-1] * 1.08)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation perplexity (log scale)")
    ax.set_title("Validation Perplexity Over Training — Small vs. Medium", fontsize=13, fontweight="bold", loc="left", color=t["ink"])
    style_axes(ax, t)

    ax.legend(frameon=False, loc="upper right", labelcolor=t["ink_secondary"], fontsize=9)

    fig.text(0.01, -0.02,
              "Perplexity = exp(validation loss), computed from the same eval passes logged during training "
              "(experiments/runs/*/logs/train_log.jsonl).",
              fontsize=8, color=t["muted"], ha="left")

    fig.tight_layout()
    out = os.path.join(HERE, f"perplexity_curves_{theme_name}.png")
    fig.savefig(out, dpi=200, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


for theme_name in THEMES:
    plot_loss_curves(theme_name)
    plot_perplexity_curves(theme_name)

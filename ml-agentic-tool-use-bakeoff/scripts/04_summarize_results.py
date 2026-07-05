#!/usr/bin/env python3
"""Aggregate all three architectures' saved metrics into a comparison figure and
a written results summary.

Usage:
    python scripts/04_summarize_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.metrics import load_metrics

ARCHITECTURES = [
    ("react", config.REACT_METRICS_PATH),
    ("plan_execute", config.PLAN_EXECUTE_METRICS_PATH),
    ("reflexion", config.REFLEXION_METRICS_PATH),
]


def load_all_metrics() -> dict[str, dict]:
    all_metrics = {}
    for name, path in ARCHITECTURES:
        if not path.exists():
            print(f"Warning: {path} not found -- run the corresponding script first. Skipping {name}.")
            continue
        all_metrics[name] = load_metrics(path)
    return all_metrics


def render_figure(all_metrics: dict[str, dict]) -> None:
    categories = sorted({cat for m in all_metrics.values() for cat in m.get("success_rate_by_category", {})})
    architectures = list(all_metrics.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    overall = [all_metrics[a]["overall_success_rate"] for a in architectures]
    axes[0].bar(architectures, overall)
    axes[0].set_title("Overall Success Rate by Architecture")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Success rate")

    x = list(range(len(categories)))
    width = 0.8 / max(len(architectures), 1)
    for i, arch in enumerate(architectures):
        rates = [all_metrics[arch]["success_rate_by_category"].get(cat, 0.0) for cat in categories]
        offset = (i - (len(architectures) - 1) / 2) * width
        axes[1].bar([xi + offset for xi in x], rates, width=width, label=arch)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories, rotation=20, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Success Rate by Category")
    axes[1].legend()

    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / "architecture_comparison.png", dpi=150)
    plt.close(fig)


def render_results_summary(all_metrics: dict[str, dict]) -> str:
    lines = ["# Results Summary", "", "## Objective", ""]
    lines.append(
        "Compare three agent architectures -- ReAct, Plan-and-Execute, and "
        "Reflexion -- on the same fixed 35-task benchmark (arithmetic, multi-hop "
        "QA, code execution, and injected-error recovery), all built directly "
        "against the OpenAI Chat Completions tool-use API with no framework "
        "abstraction."
    )
    lines += ["", "## Tasks", ""]
    lines.append(
        "35 hand-written tasks across 4 categories: 9 arithmetic, 9 multi-hop QA "
        "(over a small original synthetic knowledge base), 9 code execution, and "
        "8 error-recovery (each reusing a base task with a tool call forced to "
        "fail once). See `data/tasks.jsonl`."
    )
    lines += ["", "## Results", ""]
    lines.append(
        "| Architecture | Overall success | Mean LLM calls | Mean tool calls | "
        "Error recovery rate | Est. cost (USD) |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for arch, metrics in all_metrics.items():
        recovery = metrics.get("error_recovery_rate")
        recovery_str = f"{recovery:.2%}" if recovery is not None else "n/a"
        lines.append(
            f"| {arch} | {metrics['overall_success_rate']:.2%} | "
            f"{metrics['mean_num_llm_calls']:.2f} | {metrics['mean_num_tool_calls']:.2f} | "
            f"{recovery_str} | ${metrics['estimated_cost_usd']:.4f} |"
        )
    lines += ["", "### Success rate by category", ""]
    categories = sorted({cat for m in all_metrics.values() for cat in m.get("success_rate_by_category", {})})
    lines.append("| Architecture | " + " | ".join(categories) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in categories) + " |")
    for arch, metrics in all_metrics.items():
        row = [f"{metrics['success_rate_by_category'].get(cat, 0.0):.2%}" for cat in categories]
        lines.append(f"| {arch} | " + " | ".join(row) + " |")
    lines += [
        "",
        "> **Note on comparability:** all three architectures ran against the "
        "same model, tool set, and task set. Differences reflect architecture "
        "design, not model capability.",
        "",
        "## Interpretation",
        "",
        "_Fill in after inspecting the numbers above against `reports/*_metrics.json` "
        "-- report what actually happened, including any surprising or weak "
        "results, rather than only the flattering ones._",
        "",
        "## Key Takeaways",
        "",
        "_Fill in after review._",
        "",
        "## Future Work",
        "",
        "- Compare BM25 retrieval against a local embedding-based retriever on the "
        "same multi-hop QA tasks.",
        "- Add a second model to separate architecture effects from model-capability "
        "effects.",
        "- Expand the error-recovery category to include errors injected on the "
        "2nd or 3rd call to a tool, not just the 1st.",
    ]
    return "\n".join(lines)


def main() -> None:
    all_metrics = load_all_metrics()
    if not all_metrics:
        print("No metrics files found. Run scripts 01-03 first.")
        return

    print("=== Architecture Comparison ===")
    for arch, metrics in all_metrics.items():
        print(f"\n{arch}:")
        print(f"  Overall success rate: {metrics['overall_success_rate']:.2%}")
        print(f"  Mean LLM calls: {metrics['mean_num_llm_calls']:.2f}")
        print(f"  Mean tool calls: {metrics['mean_num_tool_calls']:.2f}")
        if metrics.get("error_recovery_rate") is not None:
            print(f"  Error recovery rate: {metrics['error_recovery_rate']:.2%}")
        print(f"  Estimated cost: ${metrics['estimated_cost_usd']:.4f}")

    render_figure(all_metrics)
    summary_md = render_results_summary(all_metrics)
    config.RESULTS_SUMMARY_PATH.write_text(summary_md)
    print(f"\nSaved comparison figure to {config.FIGURES_DIR / 'architecture_comparison.png'}")
    print(f"Saved results summary to {config.RESULTS_SUMMARY_PATH}")


if __name__ == "__main__":
    main()

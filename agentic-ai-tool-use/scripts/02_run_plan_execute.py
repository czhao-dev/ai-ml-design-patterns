#!/usr/bin/env python3
"""Run the Plan-and-Execute agent over the benchmark task set and save its metrics.

Usage:
    python scripts/02_run_plan_execute.py [--limit N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

from src import config
from src.agents.plan_execute import PlanExecuteAgent
from src.metrics import aggregate_architecture_metrics, save_metrics, score_task
from src.tasks import load_tasks
from src.tool_setup import apply_error_injection, build_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Plan-and-Execute agent over the benchmark task set.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N tasks (sanity pass).")
    args = parser.parse_args()

    settings = config.get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    registry = build_registry(settings)
    tasks = load_tasks(config.TASKS_PATH)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    agent = PlanExecuteAgent(
        client,
        settings.model_id,
        registry,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        max_steps_per_subtask=settings.max_steps_per_subtask,
    )

    eval_results = []
    for task in tasks:
        print(f"[plan_execute] {task.task_id} ({task.category})...", end=" ", flush=True)
        start = time.monotonic()
        with apply_error_injection(registry, task.error_injection):
            result = agent.run(task)
        eval_result = score_task(result, task)
        eval_results.append(eval_result)
        status = "PASS" if eval_result.success else "FAIL"
        elapsed = time.monotonic() - start
        print(f"{status} ({elapsed:.1f}s, {result.num_llm_calls} llm calls, {result.num_tool_calls} tool calls)")

    metrics = aggregate_architecture_metrics(
        eval_results, model_id=settings.model_id, pricing_table=config.MODEL_PRICING_PER_MTOK
    )
    save_metrics(metrics, config.PLAN_EXECUTE_METRICS_PATH)
    print(f"\nOverall success rate: {metrics['overall_success_rate']:.2%}")
    print(f"Estimated cost: ${metrics['estimated_cost_usd']:.4f}")
    print(f"Saved metrics to {config.PLAN_EXECUTE_METRICS_PATH}")


if __name__ == "__main__":
    main()

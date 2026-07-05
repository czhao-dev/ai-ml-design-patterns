"""Scoring and reporting for the agent-architecture bake-off.

score_task() is the single place ground truth (task.expected_answer) is ever
consulted -- agent code never sees it, only the harness that scores a
completed run.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .tasks import Task

if TYPE_CHECKING:
    from .agents.base import AgentResult


@dataclass
class TaskEvalResult:
    task_id: str
    category: str
    architecture: str
    success: bool
    predicted_answer: str | None
    expected_answer: str
    tool_precision: float | None
    tool_recall: float | None
    num_llm_calls: int
    num_tool_calls: int
    latency_sec: float
    input_tokens: int
    output_tokens: int
    error_injected: bool
    error_recovered: bool | None


def normalize_answer(answer: str, mode: str) -> str:
    if mode == "numeric":
        return answer.strip()
    if mode == "string_ci":
        return "".join(ch.lower() for ch in answer.strip() if ch.isalnum() or ch.isspace()).strip()
    if mode == "string_exact":
        return answer
    raise ValueError(f"Unknown normalization mode: {mode}")


def exact_match(predicted: str | None, expected: str, mode: str, numeric_epsilon: float = 1e-6) -> bool:
    if predicted is None:
        return False
    if mode == "numeric":
        try:
            predicted_val = float(normalize_answer(predicted, mode))
            expected_val = float(normalize_answer(expected, mode))
        except ValueError:
            return False
        return abs(predicted_val - expected_val) <= numeric_epsilon
    return normalize_answer(predicted, mode) == normalize_answer(expected, mode)


def tool_call_precision_recall(
    actual_tools: list[str], expected_tools: list[str] | None
) -> tuple[float | None, float | None]:
    """None, None if expected_tools is None (task doesn't grade tool usage).
    Otherwise treats both as sets: precision = |actual ∩ expected| / |actual|,
    recall = |actual ∩ expected| / |expected|."""
    if expected_tools is None:
        return None, None
    actual_set = set(actual_tools)
    expected_set = set(expected_tools)
    if not actual_set and not expected_set:
        return 1.0, 1.0
    intersection = len(actual_set & expected_set)
    precision = intersection / len(actual_set) if actual_set else 0.0
    recall = intersection / len(expected_set) if expected_set else 0.0
    return precision, recall


def score_task(agent_result: "AgentResult", task: Task) -> TaskEvalResult:
    success = exact_match(agent_result.final_answer, task.expected_answer, task.answer_normalization)
    actual_tools = [call["tool"] for call in agent_result.tool_calls_made]
    precision, recall = tool_call_precision_recall(actual_tools, task.expected_tools)

    error_injected = task.error_injection is not None
    error_recovered = success if error_injected else None

    return TaskEvalResult(
        task_id=task.task_id,
        category=task.category,
        architecture=agent_result.architecture,
        success=success,
        predicted_answer=agent_result.final_answer,
        expected_answer=task.expected_answer,
        tool_precision=precision,
        tool_recall=recall,
        num_llm_calls=agent_result.num_llm_calls,
        num_tool_calls=agent_result.num_tool_calls,
        latency_sec=agent_result.latency_sec,
        input_tokens=agent_result.input_tokens,
        output_tokens=agent_result.output_tokens,
        error_injected=error_injected,
        error_recovered=error_recovered,
    )


def estimate_cost_usd(
    input_tokens: int, output_tokens: int, model_id: str, pricing_table: dict[str, tuple[float, float]]
) -> float:
    if model_id not in pricing_table:
        return 0.0
    input_rate, output_rate = pricing_table[model_id]
    return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate


def aggregate_architecture_metrics(
    results: list[TaskEvalResult], model_id: str, pricing_table: dict[str, tuple[float, float]]
) -> dict:
    if not results:
        return {"n_tasks": 0}

    n = len(results)
    success_count = sum(1 for r in results if r.success)

    categories = sorted({r.category for r in results})
    success_by_category = {
        cat: sum(1 for r in results if r.category == cat and r.success)
        / sum(1 for r in results if r.category == cat)
        for cat in categories
    }

    graded = [r for r in results if r.tool_precision is not None]
    mean_tool_precision = statistics.mean(r.tool_precision for r in graded) if graded else None
    mean_tool_recall = statistics.mean(r.tool_recall for r in graded) if graded else None

    injected = [r for r in results if r.error_injected]
    error_recovery_rate = (
        sum(1 for r in injected if r.error_recovered) / len(injected) if injected else None
    )

    latencies = sorted(r.latency_sec for r in results)
    total_input_tokens = sum(r.input_tokens for r in results)
    total_output_tokens = sum(r.output_tokens for r in results)
    p95_index = max(0, int(len(latencies) * 0.95) - 1)

    return {
        "n_tasks": n,
        "model_id": model_id,
        "overall_success_rate": round(success_count / n, 4),
        "success_rate_by_category": {k: round(v, 4) for k, v in success_by_category.items()},
        "mean_tool_precision": round(mean_tool_precision, 4) if mean_tool_precision is not None else None,
        "mean_tool_recall": round(mean_tool_recall, 4) if mean_tool_recall is not None else None,
        "mean_num_llm_calls": round(statistics.mean(r.num_llm_calls for r in results), 2),
        "mean_num_tool_calls": round(statistics.mean(r.num_tool_calls for r in results), 2),
        "error_recovery_rate": round(error_recovery_rate, 4) if error_recovery_rate is not None else None,
        "latency_stats": {
            "mean": round(statistics.mean(latencies), 3),
            "median": round(statistics.median(latencies), 3),
            "p95": round(latencies[p95_index], 3),
        },
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "estimated_cost_usd": round(
            estimate_cost_usd(total_input_tokens, total_output_tokens, model_id, pricing_table), 4
        ),
        "per_task": [asdict(r) for r in results],
    }


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

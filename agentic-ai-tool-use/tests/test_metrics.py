"""Tests for src/metrics.py."""

from pathlib import Path

from src.metrics import (
    TaskEvalResult,
    aggregate_architecture_metrics,
    exact_match,
    load_metrics,
    normalize_answer,
    save_metrics,
    tool_call_precision_recall,
)


def test_normalize_answer_numeric_and_string_modes():
    assert normalize_answer(" 42.0 ", "numeric") == "42.0"
    assert normalize_answer(" Jane Okoye! ", "string_ci") == "jane okoye"
    assert normalize_answer("Exact", "string_exact") == "Exact"


def test_exact_match_numeric_epsilon_tolerance():
    assert exact_match("124.8000001", "124.8", "numeric", numeric_epsilon=1e-3)
    assert not exact_match("124.9", "124.8", "numeric", numeric_epsilon=1e-3)
    assert not exact_match(None, "124.8", "numeric")


def test_exact_match_string_ci_ignores_case_and_punctuation():
    assert exact_match("Jane Okoye.", "jane okoye", "string_ci")
    assert not exact_match("Marcus Lindqvist", "jane okoye", "string_ci")


def test_tool_precision_recall_partial_overlap():
    precision, recall = tool_call_precision_recall(["calculator", "search_knowledge_base"], ["calculator"])
    assert precision == 0.5
    assert recall == 1.0


def test_tool_precision_recall_none_when_expected_tools_is_none():
    precision, recall = tool_call_precision_recall(["calculator"], None)
    assert precision is None
    assert recall is None


def _make_result(task_id, category, success, error_injected, error_recovered) -> TaskEvalResult:
    return TaskEvalResult(
        task_id=task_id,
        category=category,
        architecture="react",
        success=success,
        predicted_answer="x",
        expected_answer="x",
        tool_precision=None,
        tool_recall=None,
        num_llm_calls=1,
        num_tool_calls=1,
        latency_sec=0.1,
        input_tokens=10,
        output_tokens=5,
        error_injected=error_injected,
        error_recovered=error_recovered,
    )


def test_error_recovery_rate_over_injected_subset_only():
    results = [
        _make_result("a", "arithmetic", True, False, None),
        _make_result("b", "error_recovery", True, True, True),
        _make_result("c", "error_recovery", False, True, False),
    ]
    metrics = aggregate_architecture_metrics(
        results, model_id="gpt-4.1-mini", pricing_table={"gpt-4.1-mini": (0.40, 1.60)}
    )
    assert metrics["error_recovery_rate"] == 0.5
    assert metrics["n_tasks"] == 3


def test_aggregate_metrics_estimates_cost():
    results = [_make_result("a", "arithmetic", True, False, None)]
    metrics = aggregate_architecture_metrics(
        results, model_id="gpt-4.1-mini", pricing_table={"gpt-4.1-mini": (0.40, 1.60)}
    )
    expected_cost = (10 / 1_000_000) * 0.40 + (5 / 1_000_000) * 1.60
    assert metrics["estimated_cost_usd"] == round(expected_cost, 4)


def test_save_and_load_metrics_roundtrip(tmp_path: Path):
    metrics = {"a": 1, "b": [1, 2, 3]}
    path = tmp_path / "sub" / "metrics.json"
    save_metrics(metrics, path)
    loaded = load_metrics(path)
    assert loaded == metrics

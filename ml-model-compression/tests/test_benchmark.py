"""Tests for src/benchmark.py."""

from __future__ import annotations

import json

import torch
import torch.nn as nn

from src import benchmark


def test_measure_latency_returns_positive_latency_and_throughput():
    model = nn.Linear(4, 2)
    result = benchmark.measure_latency(model, input_shape=(1, 4), n=5, warmup=2)

    assert result["latency_ms"] > 0
    assert result["throughput_ips"] > 0


def test_measure_model_size_from_module_and_state_dict_agree():
    model = nn.Linear(100, 100)
    size_from_module = benchmark.measure_model_size(model)
    size_from_state_dict = benchmark.measure_model_size(model.state_dict())

    assert size_from_module == size_from_state_dict
    assert size_from_module > 0


def test_measure_model_size_scales_with_parameter_count():
    small = benchmark.measure_model_size(nn.Linear(10, 10))
    big = benchmark.measure_model_size(nn.Linear(1000, 1000))
    assert big > small


def test_record_result_appends_and_upserts_by_variant(tmp_path, monkeypatch):
    cache_path = tmp_path / "results.json"
    monkeypatch.setattr(benchmark, "RESULTS_CACHE_JSON", cache_path)

    benchmark.record_result({"variant": "baseline", "accuracy": 0.9})
    benchmark.record_result({"variant": "pruned_20", "accuracy": 0.88})
    rows = json.loads(cache_path.read_text())
    assert [r["variant"] for r in rows] == ["baseline", "pruned_20"]

    # Re-recording "baseline" should replace, not duplicate, its row.
    benchmark.record_result({"variant": "baseline", "accuracy": 0.95})
    rows = json.loads(cache_path.read_text())
    assert len(rows) == 2
    baseline_row = next(r for r in rows if r["variant"] == "baseline")
    assert baseline_row["accuracy"] == 0.95


def _sample_row(**overrides):
    row = {
        "variant": "baseline",
        "technique": "Pruning",
        "accuracy": 0.9,
        "f1": 0.88,
        "size_mb": 12.5,
        "latency_ms": 3.2,
        "throughput_ips": 312.5,
        "sparsity": 0.2,
        "prune_type": "unstructured",
    }
    row.update(overrides)
    return row


def test_write_results_summary_produces_markdown_table(tmp_path, monkeypatch):
    cache_path = tmp_path / "results.json"
    summary_path = tmp_path / "results_summary.md"
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    monkeypatch.setattr(benchmark, "RESULTS_CACHE_JSON", cache_path)
    monkeypatch.setattr(benchmark, "RESULTS_SUMMARY_MD", summary_path)
    monkeypatch.setattr(benchmark, "FIGURES_DIR", figures_dir)
    cache_path.write_text(json.dumps([_sample_row()]))

    benchmark.write_results_summary()

    content = summary_path.read_text()
    assert "| baseline | Pruning | 90.00% | 0.8800 | 12.50 | 3.200 | 312.5 |" in content
    assert (figures_dir / "accuracy_vs_sparsity.png").exists()
    assert (figures_dir / "size_vs_latency.png").exists()


def test_write_results_summary_with_no_rows_writes_empty_table(tmp_path, monkeypatch):
    cache_path = tmp_path / "results.json"
    summary_path = tmp_path / "results_summary.md"
    monkeypatch.setattr(benchmark, "RESULTS_CACHE_JSON", cache_path)
    monkeypatch.setattr(benchmark, "RESULTS_SUMMARY_MD", summary_path)

    benchmark.write_results_summary()

    assert "| Model | Compression |" in summary_path.read_text()

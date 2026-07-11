#!/usr/bin/env python3
"""Benchmark: times the INT8 zero-allocation engine (Engine.forward()) against
the naive fp32 reference (fp32_reference.run_fp32_reference()) across a
batch-size sweep, on identical weights/input for each batch, and persists raw
per-batch/per-variant latency and throughput statistics to a CSV.

No plotting here -- see reports/generate_plots.py, which reads the CSV this
script writes and never re-runs anything.
"""

from __future__ import annotations

import argparse
import csv
import gc
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import demo_graph
from src import fp32_reference as ref
from src.engine import Engine


@dataclass
class LatencyStats:
    mean_ms: float
    median_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


def _time_calls(fn: Callable[[], None], iterations: int, warmup: int) -> LatencyStats:
    for _ in range(warmup):
        fn()

    samples = np.empty(iterations, dtype=np.float64)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for i in range(iterations):
            t0 = time.perf_counter()
            fn()
            samples[i] = (time.perf_counter() - t0) * 1000.0
    finally:
        if gc_was_enabled:
            gc.enable()

    return LatencyStats(
        mean_ms=float(np.mean(samples)),
        median_ms=float(np.median(samples)),
        p99_ms=float(np.percentile(samples, 99)),
        min_ms=float(np.min(samples)),
        max_ms=float(np.max(samples)),
    )


def _row(batch: int, variant: str, stats: LatencyStats, iterations: int, warmup: int,
          arena_size_bytes: int | None, host: str, python_version: str) -> dict:
    throughput = batch / (stats.mean_ms / 1000.0)
    return {
        "batch": batch,
        "variant": variant,
        "iterations": iterations,
        "warmup_iterations": warmup,
        "mean_latency_ms": f"{stats.mean_ms:.6f}",
        "median_latency_ms": f"{stats.median_ms:.6f}",
        "p99_latency_ms": f"{stats.p99_ms:.6f}",
        "min_latency_ms": f"{stats.min_ms:.6f}",
        "max_latency_ms": f"{stats.max_ms:.6f}",
        "throughput_rows_per_sec": f"{throughput:.3f}",
        "arena_size_bytes": arena_size_bytes if arena_size_bytes is not None else "",
        "host": host,
        "python_version": python_version,
    }


def benchmark_batch(batch: int, iterations: int, warmup: int, seed: int, tmp_dir: Path) -> list[dict]:
    artifact_path = str(tmp_dir / f"bench_b{batch}.tge")
    result = demo_graph.build(artifact_path, batch=batch, seed=seed)  # untimed

    host = platform.node()
    python_version = platform.python_version()

    engine = Engine(artifact_path)  # untimed: allocation #1 (file buffer) + #2 (arena)
    engine.set_input(result.calibration_input)  # untimed
    int8_stats = _time_calls(engine.forward, iterations, warmup)

    def fp32_call() -> None:
        ref.run_fp32_reference(
            result.weights, result.calibration_input, batch,
            demo_graph.INPUT_DIM, demo_graph.HIDDEN_DIM, demo_graph.OUTPUT_DIM,
        )

    fp32_stats = _time_calls(fp32_call, iterations, warmup)

    Path(artifact_path).unlink(missing_ok=True)

    return [
        _row(batch, "int8", int8_stats, iterations, warmup, result.arena_size_bytes, host, python_version),
        _row(batch, "fp32", fp32_stats, iterations, warmup, None, host, python_version),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the INT8 engine vs. the fp32 reference.")
    parser.add_argument("--batch-sizes", default="4,8,32,128,512")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="reports/benchmark_results.csv")
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    fieldnames = [
        "batch", "variant", "iterations", "warmup_iterations",
        "mean_latency_ms", "median_latency_ms", "p99_latency_ms", "min_latency_ms", "max_latency_ms",
        "throughput_rows_per_sec", "arena_size_bytes", "host", "python_version",
    ]

    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for batch in batch_sizes:
            print(f"batch={batch}: running {args.warmup} warmup + {args.iterations} timed iterations "
                  f"per variant...")
            batch_rows = benchmark_batch(batch, args.iterations, args.warmup, args.seed, tmp_dir)
            rows.extend(batch_rows)
            for row in batch_rows:
                print(f"  {row['variant']:>4}: mean={row['mean_latency_ms']}ms "
                      f"p99={row['p99_latency_ms']}ms throughput={row['throughput_rows_per_sec']} rows/s")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()

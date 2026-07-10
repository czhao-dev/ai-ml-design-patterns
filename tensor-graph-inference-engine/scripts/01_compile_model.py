#!/usr/bin/env python3
"""Offline model compiler: builds the demo graph, quantizes weights, plans
the static memory arena, and writes the compiled .tge artifact that the
runtime (src/engine.py) reads.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import demo_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile the demo TGE model artifact.")
    parser.add_argument("--output", default="models/demo.tge")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result = demo_graph.build(args.output, batch=args.batch, seed=args.seed)

    print(f"Compiled demo model -> {result.artifact_path}")
    print(f"  tensors: {result.num_tensors}")
    print(f"  nodes: {result.num_nodes}")
    print(f"  arena size: {result.arena_size_bytes} bytes")
    print(f"  weights blob size: {result.weights_blob_size_bytes} bytes")
    print(f"  batch: {args.batch}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CLI runner: loads a compiled .tge artifact via Engine, runs a forward
pass on synthetic input, and prints predictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import fp32_reference as ref
from src.engine import Engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a forward pass through a compiled TGE artifact.")
    parser.add_argument("--model", default="models/demo.tge")
    args = parser.parse_args()

    engine = Engine(args.model)
    print(f"Loaded {args.model} (arena: {engine.arena_size_bytes} bytes, batch: {engine.batch}, "
          f"input: {engine.input_size}, classes: {engine.output_classes})")

    input_ = ref.make_synthetic_input(engine.batch, engine.input_size, seed=11)
    engine.set_input(input_)
    engine.forward()

    probs = engine.probabilities()
    preds = engine.predictions()

    for row in range(engine.batch):
        formatted = " ".join(f"{p:.4f}" for p in probs[row])
        print(f"row {row}: prediction={int(preds[row])} probs=[{formatted}]")


if __name__ == "__main__":
    main()

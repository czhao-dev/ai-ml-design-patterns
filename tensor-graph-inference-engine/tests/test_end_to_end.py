"""End-to-end test: compiles the demo graph, runs it through Engine, and
compares against the fp32 reference forward pass computed over the exact
same weights/input. Also pins the measured arena size as a regression
value. Ported 1:1 from tests/test_end_to_end.cpp.
"""

from __future__ import annotations

import numpy as np

from src import demo_graph, fp32_reference as ref
from src.engine import Engine


def test_end_to_end(tmp_path):
    artifact_path = tmp_path / "end_to_end.tge"
    built = demo_graph.build(str(artifact_path), batch=8, seed=7)

    engine = Engine(str(artifact_path))
    assert engine.batch == 8
    assert engine.input_size == demo_graph.INPUT_DIM
    assert engine.output_classes == demo_graph.OUTPUT_DIM

    engine.set_input(built.calibration_input)
    engine.forward()

    probs = engine.probabilities()
    preds = engine.predictions()

    # Probabilities are a valid distribution.
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.all(np.abs(probs.sum(axis=1) - 1.0) < 1e-3)

    # Bounded error against the fp32 reference computed over the same
    # weights/input (INT8 quantization introduces error, but it must stay
    # small and predictions should usually agree).
    max_err = ref.max_abs_error(probs, built.fp32_reference.probabilities)
    mean_err = ref.mean_abs_error(probs, built.fp32_reference.probabilities)
    print(f"probability error vs fp32 reference: max={max_err:.6f} mean={mean_err:.6f}")
    assert max_err < 0.25

    agree = int(np.sum(preds == built.fp32_reference.predictions))
    print(f"predictions agreeing with fp32 reference: {agree}/{engine.batch}")
    assert agree >= engine.batch // 2

    # Regression pin: measured once at implementation time (naive sum of
    # all 15 non-weight tensors at batch=8 is 63,072 bytes; the planner
    # achieves 39,904 via lifetime-aware reuse). If this ever fails, either
    # the topology or the planner changed -- both worth a deliberate look,
    # not a silent drift.
    assert engine.arena_size_bytes == built.arena_size_bytes
    assert built.arena_size_bytes == 39904

"""Concrete demo topology: a 3-layer MLP with a residual/skip connection,
ported from include/tge/demo_graph.hpp.

  0:  Quantize(X)              -> Xq
  1:  MatmulInt8(Xq,W1,B1)     -> H1_acc
  2:  Dequantize(H1_acc)       -> H1_pre
  3:  Relu(H1_pre)             -> H1          <- residual source, lifetime [3,8]
  4:  Quantize(H1)             -> H1q
  5:  MatmulInt8(H1q,W2,B2)    -> H2_acc
  6:  Dequantize(H2_acc)       -> H2_pre
  7:  Relu(H2_pre)             -> H2_relu
  8:  Add(H2_relu,H1)          -> H2_sum       <- consumes H1 (dies here)
  9:  Quantize(H2_sum)         -> H2q
  10: MatmulInt8(H2q,W3,B3)    -> Logits_acc
  11: Dequantize(Logits_acc)   -> Logits
  12: Softmax(Logits)          -> Probs
  13: Argmax(Probs)            -> Preds

H1 must stay resident through nodes 4-7 while Xq/H1_acc/H1_pre (already dead)
get reclaimed for H1q/H2_acc/H2_pre -- this is what actually exercises
lifetime-aware arena reuse; a linear chain wouldn't.

Shared by scripts/01_compile_model.py and tests, via build(), so the compiled
artifact and the fp32 ground truth can never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import fp32_reference as ref
from src import ops
from src.arena_planner import compute_lifetimes, plan_arena
from src.fp32_reference import DemoForwardResult, DemoWeights
from src.graph import GraphBuilder
from src.model_format import write_artifact
from src.types import ALIGNMENT, TensorKind, align_up

INPUT_DIM = 784
HIDDEN_DIM = 128
OUTPUT_DIM = 10


@dataclass
class BuildResult:
    artifact_path: str
    weights: DemoWeights
    calibration_input: np.ndarray
    fp32_reference: DemoForwardResult
    num_tensors: int = 0
    num_nodes: int = 0
    arena_size_bytes: int = 0
    weights_blob_size_bytes: int = 0


def quantize_weight(values: np.ndarray, scale: float) -> np.ndarray:
    out = np.zeros(values.shape, dtype=np.int8)
    ops.op_quantize_f32_to_i8(values, out, scale)
    return out


def quantize_bias(values: np.ndarray, combined_scale: float) -> np.ndarray:
    return ops.round_half_away_from_zero(values.astype(np.float32) / np.float32(combined_scale)).astype(np.int32)


def build(output_path: str, batch: int = 8, seed: int = 7) -> BuildResult:
    weights = ref.make_synthetic_demo_weights(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, seed)
    calibration_input = ref.make_synthetic_input(batch, INPUT_DIM, seed + 4)
    fp32_result = ref.run_fp32_reference(weights, calibration_input, batch, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

    result = BuildResult(artifact_path=output_path, weights=weights, calibration_input=calibration_input,
                          fp32_reference=fp32_result)

    # Calibration: one-shot pass over the synthetic input to derive
    # per-tensor symmetric scales for every activation and weight.
    input_scale = ops.compute_symmetric_scale(calibration_input)
    w1_scale = ops.compute_symmetric_scale(weights.w1)
    w2_scale = ops.compute_symmetric_scale(weights.w2)
    w3_scale = ops.compute_symmetric_scale(weights.w3)
    hidden1_scale = ops.compute_symmetric_scale(fp32_result.h1)
    hidden2_scale = ops.compute_symmetric_scale(fp32_result.h2_sum)

    combined1_scale = input_scale * w1_scale
    combined2_scale = hidden1_scale * w2_scale
    combined3_scale = hidden2_scale * w3_scale

    g = GraphBuilder()
    x = g.add_input("X", [batch, INPUT_DIM])
    w1 = g.add_weight("W1", [INPUT_DIM, HIDDEN_DIM], quantize_weight(weights.w1, w1_scale), w1_scale)
    b1 = g.add_bias("B1", [HIDDEN_DIM], quantize_bias(weights.b1, combined1_scale))
    w2 = g.add_weight("W2", [HIDDEN_DIM, HIDDEN_DIM], quantize_weight(weights.w2, w2_scale), w2_scale)
    b2 = g.add_bias("B2", [HIDDEN_DIM], quantize_bias(weights.b2, combined2_scale))
    w3 = g.add_weight("W3", [HIDDEN_DIM, OUTPUT_DIM], quantize_weight(weights.w3, w3_scale), w3_scale)
    b3 = g.add_bias("B3", [OUTPUT_DIM], quantize_bias(weights.b3, combined3_scale))

    xq = g.add_quantize(x, input_scale, "Xq")
    h1_acc = g.add_matmul_int8(xq, w1, b1, "H1_acc")
    h1_pre = g.add_dequantize(h1_acc, combined1_scale, "H1_pre")
    h1 = g.add_relu(h1_pre, "H1")

    h1q = g.add_quantize(h1, hidden1_scale, "H1q")
    h2_acc = g.add_matmul_int8(h1q, w2, b2, "H2_acc")
    h2_pre = g.add_dequantize(h2_acc, combined2_scale, "H2_pre")
    h2_relu = g.add_relu(h2_pre, "H2_relu")
    h2_sum = g.add_add(h2_relu, h1, "H2_sum")

    h2q = g.add_quantize(h2_sum, hidden2_scale, "H2q")
    logits_acc = g.add_matmul_int8(h2q, w3, b3, "Logits_acc")
    logits = g.add_dequantize(logits_acc, combined3_scale, "Logits")
    probs = g.add_softmax(logits, "Probs")
    preds = g.add_argmax(probs, "Preds")

    g.mark_output(probs)
    g.mark_output(preds)

    lifetimes = compute_lifetimes(g.tensors(), g.nodes(), ALIGNMENT)
    arena_plan = plan_arena(lifetimes, ALIGNMENT)

    write_artifact(output_path, g.tensors(), g.nodes(), arena_plan, ALIGNMENT)

    result.num_tensors = len(g.tensors())
    result.num_nodes = len(g.nodes())
    result.arena_size_bytes = arena_plan.total_size_bytes

    weights_bytes = 0
    for t in g.tensors():
        if t.kind in (TensorKind.WEIGHT, TensorKind.BIAS):
            weights_bytes = align_up(weights_bytes, ALIGNMENT) + t.static_data.nbytes
    result.weights_blob_size_bytes = weights_bytes

    return result

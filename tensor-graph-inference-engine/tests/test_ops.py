"""Op-level unit tests: quantize/dequantize round-trip, int8 matmul vs. a
naive fp32 reference within quantization error bounds, relu, softmax,
argmax. No dependency on the graph/engine machinery. Ported 1:1 from
tests/test_ops.cpp.
"""

from __future__ import annotations

import numpy as np

from src import fp32_reference as ref
from src import ops


def test_quantize_dequantize_roundtrip():
    rng = np.random.default_rng(42)
    values = rng.uniform(-3.0, 3.0, size=256).astype(np.float32)

    scale = ops.compute_symmetric_scale(values)
    quantized = np.zeros(values.shape, dtype=np.int8)
    ops.op_quantize_f32_to_i8(values, quantized, scale)

    dequantized = np.zeros(values.shape, dtype=np.float32)
    ops.op_dequantize_i32_to_f32(quantized.astype(np.int32), dequantized, scale)

    assert np.all(np.abs(values - dequantized) <= scale + 1e-6)


def test_matmul_int8_vs_fp32_reference():
    m, k, n = 5, 37, 11  # deliberately not multiples of any block size
    rng = np.random.default_rng(7)
    act_f = rng.uniform(-1.0, 1.0, size=(m, k)).astype(np.float32)
    weight_f = rng.uniform(-0.5, 0.5, size=(k, n)).astype(np.float32)
    bias_f = rng.uniform(-0.1, 0.1, size=(n,)).astype(np.float32)

    fp32_out = ref.dense(act_f, weight_f, bias_f)

    act_scale = ops.compute_symmetric_scale(act_f)
    weight_scale = ops.compute_symmetric_scale(weight_f)
    combined_scale = act_scale * weight_scale

    act_q = np.zeros(act_f.shape, dtype=np.int8)
    ops.op_quantize_f32_to_i8(act_f, act_q, act_scale)
    weight_q = np.zeros(weight_f.shape, dtype=np.int8)
    ops.op_quantize_f32_to_i8(weight_f, weight_q, weight_scale)
    bias_q = ops.round_half_away_from_zero(bias_f / np.float32(combined_scale)).astype(np.int32)

    acc = np.zeros((m, n), dtype=np.int32)
    ops.op_matmul_int8(act_q, weight_q, bias_q, acc)
    quant_out = np.zeros((m, n), dtype=np.float32)
    ops.op_dequantize_i32_to_f32(acc, quant_out, combined_scale)

    # Bounded quantization error. Per k-term, the dominant error source is
    # rounding one operand while the other stays at near-full magnitude:
    # |a_true|*dw + |w_true|*da, bounded by max|a|*weight_scale +
    # max|w|*act_scale. Errors across k terms have effectively random
    # sign, so they accumulate like a random walk (~sqrt(k)), not linearly
    # -- hence the sqrt(k) scaling with a generous safety factor below.
    max_act = float(np.max(np.abs(act_f)))
    max_weight = float(np.max(np.abs(weight_f)))
    per_term_bound = max_act * weight_scale + max_weight * act_scale
    tolerance = per_term_bound * np.sqrt(k) * 2.0 + combined_scale

    max_err = ref.max_abs_error(fp32_out, quant_out)
    assert max_err <= tolerance


def test_relu():
    in_ = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    out = np.zeros(in_.shape, dtype=np.float32)
    ops.op_relu_f32(in_, out)
    expected = np.array([0.0, 0.0, 0.0, 0.5, 2.0], dtype=np.float32)
    assert np.array_equal(out, expected)


def test_softmax_and_argmax():
    logits = np.array([[1.0, 2.0, 0.5], [3.0, 2.9, -1.0]], dtype=np.float32)  # 2 rows x 3 cols
    probs = np.zeros(logits.shape, dtype=np.float32)
    ops.op_softmax_f32(logits, probs)

    row_sums = probs.sum(axis=1)
    assert np.all(np.abs(row_sums - 1.0) < 1e-5)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    preds = np.zeros(2, dtype=np.int32)
    ops.op_argmax_i32(probs, preds)
    assert preds[0] == 1  # logit 2.0 is max in row 0
    assert preds[1] == 0  # logit 3.0 is max in row 1

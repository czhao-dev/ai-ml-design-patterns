"""Pure compute kernels for the tensor graph engine, ported from include/tge/ops.hpp.

Every function here is the only code touched from inside Engine.forward() (see
engine.py) -- each writes its result into a caller-supplied destination array
(`out`/`dst`) rather than returning a freshly allocated one, so forward() can
write straight into a pre-planned arena view instead of allocating per step.

Quantization scheme: per-tensor symmetric INT8, zero_point is always 0.
INT8 x INT8 -> INT32 accumulation, with an optional INT32 bias fused directly
into the accumulator (the "scale-shifting vector" -- the bias vector is the
shift applied per output channel before dequantization).
"""

from __future__ import annotations

import numpy as np


def round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    """C++'s std::round/std::lround round half away from zero; NumPy's
    np.round is round-half-to-even (banker's rounding). Quantization must use
    this explicit implementation everywhere it rounds a value, or numeric-
    accuracy assertions (though not the byte-count regressions, which are
    pure shape facts) can diverge from the documented C++ semantics."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def op_quantize_f32_to_i8(src: np.ndarray, dst: np.ndarray, scale: float) -> None:
    inv_scale = np.float32(1.0) / np.float32(scale)
    scaled = round_half_away_from_zero(src.astype(np.float32) * inv_scale)
    np.clip(scaled, -128.0, 127.0, out=scaled)
    dst[...] = scaled.astype(np.int8)


def op_dequantize_i32_to_f32(src: np.ndarray, dst: np.ndarray, combined_scale: float) -> None:
    dst[...] = src.astype(np.float32) * np.float32(combined_scale)


def op_relu_f32(src: np.ndarray, dst: np.ndarray) -> None:
    np.maximum(src, np.float32(0.0), out=dst)


def op_add_f32(a: np.ndarray, b: np.ndarray, dst: np.ndarray) -> None:
    np.add(a, b, out=dst)


def op_softmax_f32(logits: np.ndarray, dst: np.ndarray) -> None:
    row_max = logits.max(axis=1, keepdims=True)
    exp = np.exp(logits - row_max)
    dst[...] = exp / exp.sum(axis=1, keepdims=True)


def op_argmax_i32(probs: np.ndarray, dst: np.ndarray) -> None:
    # np.argmax returns the first occurrence of the max, matching the C++
    # loop's strict '>' comparison (ties keep the earliest index).
    dst[...] = probs.argmax(axis=1).astype(np.int32)


def op_matmul_int8(activation: np.ndarray, weight: np.ndarray, bias: np.ndarray | None,
                    out: np.ndarray) -> None:
    """activation: [m, k] INT8. weight: [k, n] INT8. bias: [n] INT32 or None.
    out: [m, n] INT32.

    The C++ implementation is a K-blocked scalar GEMM -- a cache-locality
    optimization with no semantic effect, since it's still one sequential
    INT32 accumulator per output element. A plain int32-promoted matmul
    reproduces it exactly."""
    np.matmul(activation.astype(np.int32), weight.astype(np.int32), out=out)
    if bias is not None:
        out += bias


def compute_symmetric_scale(values: np.ndarray) -> float:
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    return max_abs / 127.0 if max_abs > 0.0 else 1.0

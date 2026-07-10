"""FP32 ground-truth forward pass and synthetic data generation, ported from
include/tge/fp32_reference.hpp.

Used offline only: to calibrate INT8 quantization scales, and as the
correctness baseline the quantized graph is checked against in
tests/test_end_to_end.py. Never touched on the hot path.

Note: this uses NumPy's PRNG (`np.random.default_rng`), not a port of C++'s
`std::mt19937`/`std::normal_distribution` bit-for-bit -- exact numeric parity
with the removed C++ implementation isn't a goal, only internal consistency
(the same seed always produces the same weights/input here), which is all
`demo_graph.build()` and the regression-pinned tests actually depend on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DemoWeights:
    w1: np.ndarray  # [input_dim, hidden_dim]
    b1: np.ndarray  # [hidden_dim]
    w2: np.ndarray  # [hidden_dim, hidden_dim]
    b2: np.ndarray  # [hidden_dim]
    w3: np.ndarray  # [hidden_dim, output_dim]
    b3: np.ndarray  # [output_dim]


@dataclass
class DemoForwardResult:
    h1: np.ndarray             # post-ReLU hidden1 (residual source), [batch, hidden_dim]
    h2_sum: np.ndarray         # post-residual-add, pre-final-relu,   [batch, hidden_dim]
    logits: np.ndarray         # [batch, output_dim]
    probabilities: np.ndarray  # [batch, output_dim]
    predictions: np.ndarray    # [batch]


def make_synthetic_demo_weights(input_dim: int, hidden_dim: int, output_dim: int, seed: int = 7) -> DemoWeights:
    rng = np.random.default_rng(seed)
    return DemoWeights(
        w1=rng.normal(0.0, 0.08, size=(input_dim, hidden_dim)).astype(np.float32),
        b1=rng.uniform(-0.02, 0.02, size=(hidden_dim,)).astype(np.float32),
        w2=rng.normal(0.0, 0.08, size=(hidden_dim, hidden_dim)).astype(np.float32),
        b2=rng.uniform(-0.02, 0.02, size=(hidden_dim,)).astype(np.float32),
        w3=rng.normal(0.0, 0.08, size=(hidden_dim, output_dim)).astype(np.float32),
        b3=rng.uniform(-0.02, 0.02, size=(output_dim,)).astype(np.float32),
    )


def make_synthetic_input(batch: int, input_dim: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(batch, input_dim)).astype(np.float32)


def dense(activation: np.ndarray, weight: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    """Naive fp32 dense layer: activation [m,k] @ weight [k,n] + bias[n] -> [m,n]."""
    out = activation.astype(np.float32) @ weight.astype(np.float32)
    if bias is not None:
        out = out + bias.astype(np.float32)
    return out.astype(np.float32)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, np.float32(0.0))


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def softmax(logits: np.ndarray) -> np.ndarray:
    row_max = logits.max(axis=1, keepdims=True)
    exp = np.exp(logits - row_max)
    return exp / exp.sum(axis=1, keepdims=True)


def argmax(probs: np.ndarray) -> np.ndarray:
    return probs.argmax(axis=1).astype(np.int32)


def run_fp32_reference(weights: DemoWeights, input_: np.ndarray, batch: int, input_dim: int,
                        hidden_dim: int, output_dim: int) -> DemoForwardResult:
    """Ground truth for the demo topology:
    dense1 -> relu -> dense2 -> relu -> add(residual with h1) -> dense3 -> softmax -> argmax
    """
    del batch, input_dim, hidden_dim, output_dim  # shapes are implied by weights/input_

    h1_pre = dense(input_, weights.w1, weights.b1)
    h1 = relu(h1_pre)

    h2_pre = dense(h1, weights.w2, weights.b2)
    h2_relu = relu(h2_pre)
    h2_sum = add(h2_relu, h1)

    logits = dense(h2_sum, weights.w3, weights.b3)
    probabilities = softmax(logits)
    predictions = argmax(probabilities)

    return DemoForwardResult(
        h1=h1, h2_sum=h2_sum, logits=logits, probabilities=probabilities, predictions=predictions,
    )


def max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def mean_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b))) if a.size else 0.0

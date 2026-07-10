"""Offline, in-memory graph representation, ported from include/tge/graph.hpp.

Used only by demo_graph.py and tests -- never by the runtime (engine.py).

Scale-field convention: every TensorInfo.scale is set exactly once, at the
moment that tensor is created, and always means "the scale needed to
interpret this tensor's values":
  - add_weight/add_bias: scale is the caller's precomputed value.
  - add_quantize(input, scale, name): the *new* I8 output tensor's
    quantization scale.
  - add_dequantize(input, combined_scale, name): the *new* F32 output
    tensor's combined_scale (input_scale * weight_scale), NOT stored on the
    I32 input. The runtime reads it from the output tensor.
No tensor's scale is ever mutated after creation.

Per-op input order convention:
  Quantize:    [input]
  MatmulInt8:  [activation, weight, bias]   (bias id may be KINVALID_ID)
  Dequantize:  [input]
  Relu:        [input]
  Add:         [a, b]
  Softmax:     [input]
  Argmax:      [input]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.types import KINVALID_ID, DType, OpType, TensorKind


@dataclass
class TensorInfo:
    id: int
    name: str
    dtype: DType
    kind: TensorKind
    dims: list[int]
    scale: float = 1.0
    zero_point: int = 0
    static_data: np.ndarray | None = None  # non-None only for Weight/Bias


@dataclass
class NodeInfo:
    index: int
    op: OpType
    inputs: list[int] = field(default_factory=list)
    output: int = KINVALID_ID


class GraphBuilder:
    def __init__(self) -> None:
        self._tensors: list[TensorInfo] = []
        self._nodes: list[NodeInfo] = []
        self._next_id = 0

    def add_input(self, name: str, dims: list[int], dtype: DType = DType.F32) -> int:
        return self._push_tensor(name, dtype, TensorKind.INPUT, dims)

    def add_weight(self, name: str, dims: list[int], data: np.ndarray, scale: float) -> int:
        return self._push_tensor(name, DType.I8, TensorKind.WEIGHT, dims, scale=scale,
                                  static_data=np.asarray(data, dtype=np.int8))

    def add_bias(self, name: str, dims: list[int], data: np.ndarray) -> int:
        return self._push_tensor(name, DType.I32, TensorKind.BIAS, dims,
                                  static_data=np.asarray(data, dtype=np.int32))

    def add_quantize(self, input_: int, scale: float, name: str) -> int:
        out_id = self._new_activation(name, list(self.tensor(input_).dims), DType.I8, scale)
        self._push_node(OpType.QUANTIZE, [input_], out_id)
        return out_id

    def add_matmul_int8(self, act: int, weight: int, bias: int, name: str) -> int:
        a, w = self.tensor(act), self.tensor(weight)
        out_dims = [a.dims[0], w.dims[1]]
        out_id = self._new_activation(name, out_dims, DType.I32, 1.0)
        self._push_node(OpType.MATMUL_INT8, [act, weight, bias], out_id)
        return out_id

    def add_dequantize(self, input_: int, combined_scale: float, name: str) -> int:
        out_id = self._new_activation(name, list(self.tensor(input_).dims), DType.F32, combined_scale)
        self._push_node(OpType.DEQUANTIZE, [input_], out_id)
        return out_id

    def add_relu(self, input_: int, name: str) -> int:
        out_id = self._new_activation(name, list(self.tensor(input_).dims), DType.F32, 1.0)
        self._push_node(OpType.RELU, [input_], out_id)
        return out_id

    def add_add(self, a: int, b: int, name: str) -> int:
        out_id = self._new_activation(name, list(self.tensor(a).dims), DType.F32, 1.0)
        self._push_node(OpType.ADD, [a, b], out_id)
        return out_id

    def add_softmax(self, input_: int, name: str) -> int:
        out_id = self._new_activation(name, list(self.tensor(input_).dims), DType.F32, 1.0)
        self._push_node(OpType.SOFTMAX, [input_], out_id)
        return out_id

    def add_argmax(self, input_: int, name: str) -> int:
        out_dims = [self.tensor(input_).dims[0]]
        out_id = self._new_activation(name, out_dims, DType.I32, 1.0)
        self._push_node(OpType.ARGMAX, [input_], out_id)
        return out_id

    def mark_output(self, tensor_id: int) -> None:
        self._tensors[tensor_id].kind = TensorKind.OUTPUT

    def tensors(self) -> list[TensorInfo]:
        return self._tensors

    def nodes(self) -> list[NodeInfo]:
        return self._nodes

    def tensor(self, tensor_id: int) -> TensorInfo:
        return self._tensors[tensor_id]

    def _new_activation(self, name: str, dims: list[int], dtype: DType, scale: float) -> int:
        return self._push_tensor(name, dtype, TensorKind.ACTIVATION, dims, scale=scale)

    def _push_tensor(self, name: str, dtype: DType, kind: TensorKind, dims: list[int],
                      scale: float = 1.0, static_data: np.ndarray | None = None) -> int:
        tensor_id = self._next_id
        self._next_id += 1
        self._tensors.append(TensorInfo(id=tensor_id, name=name, dtype=dtype, kind=kind,
                                         dims=list(dims), scale=scale, static_data=static_data))
        return tensor_id

    def _push_node(self, op: OpType, inputs: list[int], output: int) -> None:
        self._nodes.append(NodeInfo(index=len(self._nodes), op=op, inputs=list(inputs), output=output))

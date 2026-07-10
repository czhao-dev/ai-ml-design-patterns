"""The zero-allocation-spirit runtime, ported from include/tge/engine.hpp.

Engine.__init__ (via _load) is the only place this class may allocate: it
reads the compiled artifact into a buffer (allocation #1) and allocates a
zero-initialized arena sized from the artifact header (allocation #2).
Engine.forward() thereafter only takes basic-slice + .view(dtype) views into
those two buffers (both zero-copy in NumPy) and calls into ops.py -- no new
arrays are allocated on the hot path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src import model_format, ops
from src.types import KINVALID_ID, NUMPY_DTYPE, DType, OpType, TensorKind


class Engine:
    def __init__(self, artifact_path: str) -> None:
        self._file_buffer: bytes = Path(artifact_path).read_bytes()  # allocation #1
        self._view = model_format.parse(self._file_buffer)
        self._arena = np.zeros(int(self._view.header["arena_size_bytes"]), dtype=np.uint8)  # allocation #2

        self._input_id = KINVALID_ID
        self._probs_id = KINVALID_ID
        self._preds_id = KINVALID_ID
        for desc in self._view.tensors:
            kind = TensorKind(int(desc["kind"]))
            if kind == TensorKind.INPUT:
                self._input_id = int(desc["id"])
            elif kind == TensorKind.OUTPUT:
                # The demo graph marks exactly two outputs: probabilities
                # (F32) and predictions (I32). Distinguish by dtype.
                if DType(int(desc["dtype"])) == DType.F32:
                    self._probs_id = int(desc["id"])
                else:
                    self._preds_id = int(desc["id"])

        if KINVALID_ID in (self._input_id, self._probs_id, self._preds_id):
            raise ValueError("tge: artifact missing required input/output tensors")

    def set_input(self, data: np.ndarray) -> None:
        self._tensor_view(self._view.tensors[self._input_id])[...] = data

    def forward(self) -> None:
        for node in self._view.nodes:
            self._run_node(node)

    def probabilities(self) -> np.ndarray:
        return self._tensor_view(self._view.tensors[self._probs_id])

    def predictions(self) -> np.ndarray:
        return self._tensor_view(self._view.tensors[self._preds_id])

    @property
    def arena_size_bytes(self) -> int:
        return int(self._view.header["arena_size_bytes"])

    @property
    def batch(self) -> int:
        return int(self._view.tensors[self._input_id]["dims"][0])

    @property
    def input_size(self) -> int:
        return int(self._view.tensors[self._input_id]["dims"][1])

    @property
    def output_classes(self) -> int:
        return int(self._view.tensors[self._probs_id]["dims"][1])

    def _tensor_view(self, desc: np.void) -> np.ndarray:
        kind = TensorKind(int(desc["kind"]))
        np_dtype = NUMPY_DTYPE[DType(int(desc["dtype"]))]
        rank = int(desc["rank"])
        dims = tuple(int(d) for d in desc["dims"][:rank])
        nbytes = int(desc["size_bytes"])
        offset = int(desc["offset"])

        base = self._view.weights_blob if kind in (TensorKind.WEIGHT, TensorKind.BIAS) else self._arena
        # Basic slice + .view(dtype): both zero-copy in NumPy, the exact
        # analogue of `arena_.data() + desc.offset` pointer arithmetic.
        return base[offset:offset + nbytes].view(np_dtype).reshape(dims)

    def _run_node(self, node: np.void) -> None:
        op = OpType(int(node["op_type"]))
        out = self._tensor_view(self._view.tensors[node["output_id"]])
        out_desc = self._view.tensors[node["output_id"]]

        if op == OpType.QUANTIZE:
            src = self._tensor_view(self._view.tensors[node["input_ids"][0]])
            ops.op_quantize_f32_to_i8(src, out, float(out_desc["scale"]))
        elif op == OpType.MATMUL_INT8:
            act = self._tensor_view(self._view.tensors[node["input_ids"][0]])
            weight = self._tensor_view(self._view.tensors[node["input_ids"][1]])
            bias_id = int(node["input_ids"][2])
            bias = self._tensor_view(self._view.tensors[bias_id]) if bias_id != KINVALID_ID else None
            ops.op_matmul_int8(act, weight, bias, out)
        elif op == OpType.DEQUANTIZE:
            src = self._tensor_view(self._view.tensors[node["input_ids"][0]])
            ops.op_dequantize_i32_to_f32(src, out, float(out_desc["scale"]))
        elif op == OpType.RELU:
            src = self._tensor_view(self._view.tensors[node["input_ids"][0]])
            ops.op_relu_f32(src, out)
        elif op == OpType.ADD:
            a = self._tensor_view(self._view.tensors[node["input_ids"][0]])
            b = self._tensor_view(self._view.tensors[node["input_ids"][1]])
            ops.op_add_f32(a, b, out)
        elif op == OpType.SOFTMAX:
            src = self._tensor_view(self._view.tensors[node["input_ids"][0]])
            ops.op_softmax_f32(src, out)
        elif op == OpType.ARGMAX:
            src = self._tensor_view(self._view.tensors[node["input_ids"][0]])
            ops.op_argmax_i32(src, out)

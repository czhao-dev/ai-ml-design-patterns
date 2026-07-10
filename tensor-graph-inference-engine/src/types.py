"""Core tensor types and alignment helpers, ported from include/tge/types.hpp.

Enum values are part of the on-disk artifact format (see model_format.py) and
must not be renumbered.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class DType(IntEnum):
    F32 = 0
    I8 = 1
    I32 = 2


class TensorKind(IntEnum):
    ACTIVATION = 0
    WEIGHT = 1
    BIAS = 2
    INPUT = 3
    OUTPUT = 4


class OpType(IntEnum):
    QUANTIZE = 0
    MATMUL_INT8 = 1
    DEQUANTIZE = 2
    RELU = 3
    ADD = 4
    SOFTMAX = 5
    ARGMAX = 6


KINVALID_ID = 0xFFFFFFFF
ALIGNMENT = 16
MAX_DIMS = 4
MAX_NODE_INPUTS = 3

_DTYPE_SIZE = {DType.F32: 4, DType.I8: 1, DType.I32: 4}
NUMPY_DTYPE = {DType.F32: np.float32, DType.I8: np.int8, DType.I32: np.int32}


def dtype_size(dtype: DType) -> int:
    return _DTYPE_SIZE[dtype]


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment

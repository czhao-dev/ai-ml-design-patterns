"""Byte-precise compiled model artifact ("TGEM" -- Tensor Graph Engine
Model), ported from include/tge/model_format.hpp.

Layout, back to back with no gaps:

  [0                                  ) FileHeader                (64 bytes)
  [64                                 ) TensorDesc[num_tensors]    (48 bytes each)
  [64 + 48*num_tensors                ) NodeDesc[num_nodes]        (24 bytes each)
  [64 + 48*num_tensors + 24*num_nodes ) weights_blob                (weights_blob_size_bytes)

The arena's *bytes* are never stored on disk, only its *size*
(arena_size_bytes) -- the runtime allocates a fresh zero-initialized buffer
of that size once at load time (see engine.py).

parse() is zero-copy: np.frombuffer over an immutable bytes object always
returns a read-only view, never a copy -- the exact analogue of the C++
reinterpret_cast-based zero-copy parse(). write_artifact() is offline-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.arena_planner import ArenaPlan
from src.graph import NodeInfo, TensorInfo
from src.types import KINVALID_ID, MAX_DIMS, MAX_NODE_INPUTS, TensorKind, align_up, dtype_size

MAGIC = b"TGEM"
FORMAT_VERSION = 1

FILE_HEADER_DTYPE = np.dtype([
    ("magic", "S4"),
    ("version", "<u4"),
    ("num_tensors", "<u4"),
    ("num_nodes", "<u4"),
    ("arena_size_bytes", "<u4"),
    ("weights_blob_size_bytes", "<u4"),
    ("alignment", "<u4"),
    ("reserved", "<u4", (9,)),
])
assert FILE_HEADER_DTYPE.itemsize == 64, "FileHeader must be exactly 64 bytes"

TENSOR_DESC_DTYPE = np.dtype([
    ("id", "<u4"),
    ("dtype", "<u4"),
    ("kind", "<u4"),
    ("rank", "<u4"),
    ("dims", "<u4", (MAX_DIMS,)),
    ("offset", "<u4"),
    ("size_bytes", "<u4"),
    ("scale", "<f4"),
    ("zero_point", "<i4"),
])
assert TENSOR_DESC_DTYPE.itemsize == 48, "TensorDesc must be exactly 48 bytes"

NODE_DESC_DTYPE = np.dtype([
    ("op_type", "<u4"),
    ("input_ids", "<u4", (MAX_NODE_INPUTS,)),
    ("output_id", "<u4"),
    ("reserved", "<u4"),
])
assert NODE_DESC_DTYPE.itemsize == 24, "NodeDesc must be exactly 24 bytes"


@dataclass
class ArtifactView:
    header: np.void
    tensors: np.ndarray
    nodes: np.ndarray
    weights_blob: np.ndarray


def parse(buffer: bytes) -> ArtifactView:
    if len(buffer) < FILE_HEADER_DTYPE.itemsize:
        raise ValueError("tge: artifact too small for header")

    header = np.frombuffer(buffer, dtype=FILE_HEADER_DTYPE, count=1)[0]
    if bytes(header["magic"]) != MAGIC:
        raise ValueError("tge: bad magic in artifact")
    if int(header["version"]) != FORMAT_VERSION:
        raise ValueError("tge: unsupported artifact version")

    num_tensors, num_nodes = int(header["num_tensors"]), int(header["num_nodes"])
    tensors_offset = FILE_HEADER_DTYPE.itemsize
    nodes_offset = tensors_offset + TENSOR_DESC_DTYPE.itemsize * num_tensors
    weights_offset = nodes_offset + NODE_DESC_DTYPE.itemsize * num_nodes
    expected_size = weights_offset + int(header["weights_blob_size_bytes"])
    if len(buffer) < expected_size:
        raise ValueError("tge: artifact truncated")

    tensors = np.frombuffer(buffer, dtype=TENSOR_DESC_DTYPE, count=num_tensors, offset=tensors_offset)
    nodes = np.frombuffer(buffer, dtype=NODE_DESC_DTYPE, count=num_nodes, offset=nodes_offset)
    weights_blob = np.frombuffer(buffer, dtype=np.uint8,
                                  count=int(header["weights_blob_size_bytes"]), offset=weights_offset)
    return ArtifactView(header=header, tensors=tensors, nodes=nodes, weights_blob=weights_blob)


def write_artifact(path: str, tensors: list[TensorInfo], nodes: list[NodeInfo],
                    arena_plan: ArenaPlan, alignment: int) -> None:
    """Offline only: packs Weight/Bias tensors sequentially (with alignment,
    never reused) into a weights blob, then writes the full artifact."""
    offset_by_tensor_id = {a.tensor_id: a.offset for a in arena_plan.assignments}

    tensor_descs = np.zeros(len(tensors), dtype=TENSOR_DESC_DTYPE)
    weights_blob = bytearray()
    weights_high_water = 0

    for t in tensors:
        dims = list(t.dims) + [1] * (MAX_DIMS - len(t.dims))
        desc = tensor_descs[t.id]
        desc["id"] = t.id
        desc["dtype"] = int(t.dtype)
        desc["kind"] = int(t.kind)
        desc["rank"] = len(t.dims)
        desc["dims"] = dims[:MAX_DIMS]
        desc["scale"] = t.scale
        desc["zero_point"] = t.zero_point

        if t.kind in (TensorKind.WEIGHT, TensorKind.BIAS):
            data_bytes = t.static_data.tobytes()
            desc["size_bytes"] = len(data_bytes)
            desc["offset"] = align_up(weights_high_water, alignment)
            pad = desc["offset"] - len(weights_blob)
            weights_blob.extend(b"\x00" * pad)
            weights_blob.extend(data_bytes)
            weights_high_water = int(desc["offset"]) + len(data_bytes)
        else:
            desc["offset"] = offset_by_tensor_id.get(t.id, KINVALID_ID)
            count = 1
            for d in t.dims:
                count *= d
            desc["size_bytes"] = align_up(count * dtype_size(t.dtype), alignment)

    node_descs = np.zeros(len(nodes), dtype=NODE_DESC_DTYPE)
    for n in nodes:
        desc = node_descs[n.index]
        desc["op_type"] = int(n.op)
        input_ids = list(n.inputs) + [KINVALID_ID] * (MAX_NODE_INPUTS - len(n.inputs))
        desc["input_ids"] = input_ids[:MAX_NODE_INPUTS]
        desc["output_id"] = n.output

    header = np.zeros(1, dtype=FILE_HEADER_DTYPE)[0]
    header["magic"] = MAGIC
    header["version"] = FORMAT_VERSION
    header["num_tensors"] = len(tensor_descs)
    header["num_nodes"] = len(node_descs)
    header["arena_size_bytes"] = arena_plan.total_size_bytes
    header["weights_blob_size_bytes"] = len(weights_blob)
    header["alignment"] = alignment

    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tensor_descs.tobytes())
        f.write(node_descs.tobytes())
        f.write(bytes(weights_blob))

"""Offline static memory planner, ported from include/tge/arena_planner.hpp.

The centerpiece of the project. Given a topologically-sorted node list,
compute the exact lifetime of every intermediate activation tensor (the range
of node indices during which it must remain valid), then run a greedy
interval/linear-scan allocator to assign byte offsets within a single
contiguous arena -- reusing space for tensors whose lifetimes don't overlap.

Weight/Bias tensors are NOT planned here: they're placed by a trivial
sequential bump-allocator into a separate, never-reused weights blob (see
model_format.py's write_artifact). Only Activation/Input/Output tensors
participate in arena planning.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.graph import NodeInfo, TensorInfo
from src.types import TensorKind, align_up, dtype_size


@dataclass
class TensorLifetime:
    tensor_id: int
    start_node: int
    end_node: int
    size_bytes: int
    pinned: bool


@dataclass
class ArenaAssignment:
    tensor_id: int
    offset: int


@dataclass
class ArenaPlan:
    assignments: list[ArenaAssignment]
    total_size_bytes: int


def compute_lifetimes(tensors: list[TensorInfo], nodes: list[NodeInfo], alignment: int) -> list[TensorLifetime]:
    result: list[TensorLifetime] = []
    last_node = len(nodes) - 1

    for t in tensors:
        if t.kind not in (TensorKind.ACTIVATION, TensorKind.INPUT, TensorKind.OUTPUT):
            continue

        count = 1
        for d in t.dims:
            count *= d
        size_bytes = align_up(count * dtype_size(t.dtype), alignment)

        if t.kind in (TensorKind.INPUT, TensorKind.OUTPUT):
            start_node, end_node, pinned = 0, last_node, True
        else:
            pinned = False
            start_node = next(n.index for n in nodes if n.output == t.id)
            consumers = [n.index for n in nodes if t.id in n.inputs]
            end_node = max(consumers) if consumers else last_node

        result.append(TensorLifetime(tensor_id=t.id, start_node=start_node, end_node=end_node,
                                      size_bytes=size_bytes, pinned=pinned))

    return result


def plan_arena(lifetimes: list[TensorLifetime], alignment: int) -> ArenaPlan:
    lifetimes = sorted(lifetimes, key=lambda life: (life.start_node, life.tensor_id))

    free_list: list[list[int]] = []  # [offset, size], checked in insertion order (first-fit)
    active: list[dict] = []          # {tensor_id, end_node, offset, size, pinned}
    high_water = 0
    assignments: list[ArenaAssignment] = []

    for life in lifetimes:
        # Expire non-pinned blocks that are provably dead before this
        # tensor's lifetime starts. Strict '<' (not '<='): a tensor
        # produced/consumed at the same node boundary as another is never
        # aliased mid-computation.
        still_active = []
        for a in active:
            if not a["pinned"] and a["end_node"] < life.start_node:
                free_list.append([a["offset"], a["size"]])
            else:
                still_active.append(a)
        active = still_active

        # First-fit allocation from the free list.
        offset = None
        for i, block in enumerate(free_list):
            if block[1] >= life.size_bytes:
                offset = block[0]
                remainder_offset, remainder_size = block[0] + life.size_bytes, block[1] - life.size_bytes
                del free_list[i]
                if remainder_size > 0:
                    free_list.append([remainder_offset, remainder_size])
                break
        if offset is None:
            offset = align_up(high_water, alignment)
            high_water = offset + life.size_bytes

        assignments.append(ArenaAssignment(tensor_id=life.tensor_id, offset=offset))
        active.append({"tensor_id": life.tensor_id, "end_node": life.end_node, "offset": offset,
                        "size": life.size_bytes, "pinned": life.pinned})

    return ArenaPlan(assignments=assignments, total_size_bytes=align_up(high_water, alignment))

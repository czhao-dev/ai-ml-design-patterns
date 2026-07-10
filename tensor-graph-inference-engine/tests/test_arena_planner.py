"""Arena planner correctness tests: synthetic DAGs with known
overlapping/disjoint lifetimes, asserting exact offsets/sizes, plus a
regression pin against the real demo graph. Ported 1:1 from
tests/test_arena_planner.cpp.
"""

from __future__ import annotations

from src import demo_graph
from src.arena_planner import TensorLifetime, plan_arena


def _offsets_by_tensor_id(plan) -> dict[int, int]:
    return {a.tensor_id: a.offset for a in plan.assignments}


def test_disjoint_lifetimes_reuse_same_offset():
    """Two tensors with strictly disjoint lifetimes and equal size must be
    assigned the SAME offset (proves reuse happened)."""
    lifetimes = [
        TensorLifetime(tensor_id=0, start_node=0, end_node=1, size_bytes=64, pinned=False),
        TensorLifetime(tensor_id=1, start_node=2, end_node=3, size_bytes=64, pinned=False),
    ]
    plan = plan_arena(lifetimes, 16)
    assert len(plan.assignments) == 2
    offsets = _offsets_by_tensor_id(plan)
    assert offsets[0] == offsets[1]
    assert plan.total_size_bytes == 64


def test_overlapping_lifetimes_disjoint_offsets():
    """Two tensors with overlapping lifetimes must get non-intersecting byte
    ranges."""
    lifetimes = [
        TensorLifetime(tensor_id=0, start_node=0, end_node=5, size_bytes=64, pinned=False),
        TensorLifetime(tensor_id=1, start_node=2, end_node=3, size_bytes=128, pinned=False),
    ]
    plan = plan_arena(lifetimes, 16)
    offsets = _offsets_by_tensor_id(plan)
    off0, off1 = offsets[0], offsets[1]
    assert (off0 + 64 <= off1) or (off1 + 128 <= off0)
    assert plan.total_size_bytes == 64 + 128


def test_pinned_tensor_never_reused():
    """A pinned tensor is never reused and spans the whole graph; its
    (forced) end_node must not cause it to be offered to the free list."""
    lifetimes = [
        TensorLifetime(tensor_id=0, start_node=0, end_node=9, size_bytes=64, pinned=True),   # pinned, spans graph
        TensorLifetime(tensor_id=1, start_node=0, end_node=1, size_bytes=64, pinned=False),  # dies early
        TensorLifetime(tensor_id=2, start_node=2, end_node=9, size_bytes=64, pinned=False),  # could reuse tensor 1
    ]
    plan = plan_arena(lifetimes, 16)
    offsets = _offsets_by_tensor_id(plan)
    assert offsets[1] == offsets[2]  # tensor 1's slot got reused by tensor 2
    assert offsets[0] != offsets[1]  # pinned tensor 0 never touched


def test_exact_total_with_known_overlap():
    """Hand-constructed case with known round-number sizes: assert the EXACT
    expected total, not just "less than the naive sum".

    Lifetimes: A[0,1] sz64, B[1,2] sz64, C[2,3] sz64 (reuses A's slot).
    A and B overlap at node 1 (A end=1, B start=1 -> not expired, "1<1" is
    false). B and C overlap at node 2 similarly, so C must reuse A's slot.
    """
    lifetimes = [
        TensorLifetime(tensor_id=0, start_node=0, end_node=1, size_bytes=64, pinned=False),
        TensorLifetime(tensor_id=1, start_node=1, end_node=2, size_bytes=64, pinned=False),
        TensorLifetime(tensor_id=2, start_node=2, end_node=3, size_bytes=64, pinned=False),
    ]
    plan = plan_arena(lifetimes, 16)
    offsets = _offsets_by_tensor_id(plan)
    assert offsets[0] != offsets[1]
    assert offsets[1] != offsets[2]
    assert offsets[0] == offsets[2]
    assert plan.total_size_bytes == 128  # exactly 2 concurrent slots needed, not 3


def test_demo_graph_arena_regression(tmp_path):
    """Regression pin against the real demo graph: measure once, hard-code
    the observed value so future changes to the planner or topology that
    regress the arena size are caught."""
    result = demo_graph.build(str(tmp_path / "arena_regression.tge"), batch=8, seed=7)
    assert result.num_tensors == 21
    assert result.num_nodes == 14
    # Naive sum of all 15 non-weight tensors at batch=8 is 63,072 bytes.
    # Measured once at implementation time: 39,904 bytes (a ~37% cut from
    # lifetime-aware reuse). Hard-coded so a future change to the planner
    # or topology that regresses the arena size gets caught deliberately.
    assert result.arena_size_bytes == 39904

"""Reinterpretation of tests/test_zero_alloc.cpp for Python.

The original test proved the project's central claim by overriding global
operator new/delete and asserting zero heap allocations across repeated
forward() calls. CPython/NumPy can't literally intercept malloc the same
way -- numpy ufuncs and `.astype()` casts still touch small transient
C-level temporaries per call, which is out of scope to eliminate without
bypassing numpy's ufunc machinery entirely.

Instead, this file tests the two properties that actually capture the
"zero-allocation-spirit" runtime design:
  1. The arena buffer is allocated exactly once (at Engine construction) and
     never reallocated or rebound by forward()/set_input().
  2. forward()'s outputs are views into that same arena (`np.shares_memory`),
     not freshly allocated arrays -- and repeated forward() calls do not
     grow Python-tracked memory over time.
"""

from __future__ import annotations

import tracemalloc

import numpy as np

from src import demo_graph, fp32_reference
from src.engine import Engine


def _build_engine(tmp_path):
    artifact_path = tmp_path / "zero_alloc.tge"
    demo_graph.build(str(artifact_path), batch=8, seed=7)
    return Engine(str(artifact_path))


def test_arena_allocated_only_at_construction(tmp_path):
    engine = _build_engine(tmp_path)

    arena_id_before = id(engine._arena)
    arena_ptr_before = engine._arena.__array_interface__["data"][0]

    input_ = fp32_reference.make_synthetic_input(engine.batch, engine.input_size, seed=99)
    for _ in range(5):
        engine.set_input(input_)
        engine.forward()

    assert id(engine._arena) == arena_id_before
    assert engine._arena.__array_interface__["data"][0] == arena_ptr_before


def test_output_views_share_arena_memory(tmp_path):
    engine = _build_engine(tmp_path)
    input_ = fp32_reference.make_synthetic_input(engine.batch, engine.input_size, seed=99)
    engine.set_input(input_)
    engine.forward()

    assert np.shares_memory(engine.probabilities(), engine._arena)
    assert np.shares_memory(engine.predictions(), engine._arena)


def test_forward_has_no_tensor_scale_allocation_growth(tmp_path):
    """Repeated forward() calls must not allocate a new tensor's worth of
    memory every call -- that would mean the arena-reuse property broke
    down and forward() started allocating fresh arrays instead of writing
    into pre-planned views.

    This does NOT assert literal zero growth: even after a long warm-up,
    NumPy's ufunc dispatch machinery and CPython's own small-object
    allocator retain on the order of tens of bytes per call (measured:
    ~30-55 bytes/call here), which is real but has nothing to do with
    tensor data -- it's Python/NumPy call overhead, not a leak in the
    engine. The bound below (one largest-tensor's worth of bytes, X at
    [8, 784] f32 = 25,088 bytes) is what would actually indicate the
    "zero-allocation" design property regressed.
    """
    engine = _build_engine(tmp_path)
    input_ = fp32_reference.make_synthetic_input(engine.batch, engine.input_size, seed=99)
    engine.set_input(input_)
    for _ in range(10):
        engine.forward()  # warm up: let ufunc dispatch/allocator caches settle

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    for _ in range(20):
        engine.forward()
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    growth = sum(stat.size_diff for stat in snapshot_after.compare_to(snapshot_before, "lineno"))
    largest_tensor_bytes = 8 * 784 * 4  # X, [batch, input_dim] f32
    assert growth < largest_tensor_bytes

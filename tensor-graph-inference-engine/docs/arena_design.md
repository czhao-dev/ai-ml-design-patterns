# Arena Design Notes

This document is the deep-dive companion to the README's summary of the static memory planner in `src/arena_planner.py`. It covers the full algorithm and a hand-traced worked example against the actual demo topology.

## Why lifetimes, not just a bump allocator

A naive engine could just bump-allocate every intermediate tensor and never free anything — simple, but memory usage grows with the *total number* of tensors in the graph, not the number that are *simultaneously alive*. For deep or wide networks that's wasteful, and it's exactly the kind of thing an offline compiler can do better than a runtime ever could, because the compiler can see the entire computation in advance.

The key insight: once a node has consumed a tensor for the last time, that tensor's memory can be handed to a different tensor produced later — as long as their lifetimes don't overlap. Computing "does not overlap" precisely, for every pair of tensors in the graph, is the whole job of this planner.

## Step 1: Computing lifetimes (`compute_lifetimes`)

Given a topologically-sorted node list (nodes numbered `0..N-1` in execution order) and the tensor table:

- Only `Activation`, `Input`, and `Output` tensors get a lifetime. `Weight`/`Bias` tensors are placed separately (see "Weights blob" below) — they're never reused, so there's nothing to plan.
- For an **Activation** tensor: `start_node` is the index of the node that produces it (there is exactly one). `end_node` is the *maximum* index over all nodes that consume it as an input. If nothing consumes it (shouldn't happen for a well-formed graph, but handled defensively), `end_node` defaults to the last node.
- For an **Input** or **Output** tensor: forced `pinned = true`, `start_node = 0`, `end_node = N-1`. The caller writes the input before node 0 runs and reads the output after the last node runs, so these must never be reclaimed or aliased with anything else.

```
function compute_lifetimes(tensors, nodes, alignment):
    result = []
    for t in tensors where t.kind in {Activation, Input, Output}:
        if t.kind == Input or t.kind == Output:
            start, end, pinned = 0, len(nodes) - 1, true
        else:
            start = index of the node whose output == t.id
            consumers = [n.index for n in nodes if t.id in n.inputs]
            end = max(consumers) if consumers else len(nodes) - 1
            pinned = false
        size = align_up(product(t.dims) * dtype_size(t.dtype), alignment)
        result.append({t.id, start, end, size, pinned})
    return result
```

## Step 2: Planning the arena (`plan_arena`)

A greedy first-fit interval/linear-scan allocator — the same family of algorithm used for register allocation in compilers:

```
function plan_arena(lifetimes, alignment):
    sort lifetimes by (start_node asc, tensor_id asc)
    free_list = []      # {offset, size}, checked in insertion order (first-fit)
    active = []         # {tensor_id, end_node, offset, size, pinned}
    high_water = 0
    assignments = []

    for L in lifetimes:
        # Expire non-pinned blocks that are provably dead before L starts.
        # Strict '<' (not '<='): a tensor produced/consumed at the same
        # node boundary as another is never aliased mid-computation.
        for A in list(active):
            if not A.pinned and A.end_node < L.start_node:
                free_list.append({A.offset, A.size})
                active.remove(A)

        chosen = first block in free_list with size >= L.size_bytes
        if chosen exists:
            free_list.remove(chosen)
            if chosen.size > L.size_bytes:
                free_list.append({chosen.offset + L.size_bytes, chosen.size - L.size_bytes})
            offset = chosen.offset
        else:
            offset = align_up(high_water, alignment)
            high_water = offset + L.size_bytes

        assignments.append({L.tensor_id, offset})
        active.append({L.tensor_id, L.end_node, offset, L.size_bytes, L.pinned})

    return ArenaPlan{assignments, align_up(high_water, alignment)}
```

Correctness properties this must satisfy (and that `tests/test_arena_planner.py` checks directly):

1. Two tensors with strictly disjoint lifetimes and equal size get the **same offset** (reuse happened).
2. Two tensors with overlapping lifetimes get **non-intersecting** byte ranges — never aliased.
3. A pinned tensor's offset is **never** reused by anything else, regardless of any other tensor's lifetime.
4. The strict `<` expiry boundary means a tensor produced by the same node that consumes another tensor for the last time is treated as *overlapping*, not disjoint — both must be live simultaneously during that node's execution.

## Weights blob (not planned)

`Weight`/`Bias` tensors don't participate in the interval allocator at all: `model_format.py`'s `write_artifact()` places them with a trivial sequential bump-allocator (respecting alignment) into a separate region of the compiled file, once, at compile time. They're read-only for the lifetime of the `Engine` and never reused, so there's nothing to plan — planning only matters for tensors whose lifetimes actually end.

## Worked example: the demo graph

The demo topology (`src/demo_graph.py`) is a 3-layer MLP with a residual connection, `batch=8, input_dim=784, hidden_dim=128, output_dim=10`. Its 14 nodes produce these activation/input/output tensors (weights/biases excluded — they live in the separate blob):

| tensor | dtype | shape | bytes | start | end | pinned |
|---|---|---|---|---|---|---|
| X (input) | F32 | [8,784] | 25088 | 0 | 13 | yes |
| Xq | I8 | [8,784] | 6272 | 0 | 1 | no |
| H1_acc | I32 | [8,128] | 4096 | 1 | 2 | no |
| H1_pre | F32 | [8,128] | 4096 | 2 | 3 | no |
| **H1** | F32 | [8,128] | 4096 | **3** | **8** | no |
| H1q | I8 | [8,128] | 1024 | 4 | 5 | no |
| H2_acc | I32 | [8,128] | 4096 | 5 | 6 | no |
| H2_pre | F32 | [8,128] | 4096 | 6 | 7 | no |
| H2_relu | F32 | [8,128] | 4096 | 7 | 8 | no |
| H2_sum | F32 | [8,128] | 4096 | 8 | 9 | no |
| H2q | I8 | [8,128] | 1024 | 9 | 10 | no |
| Logits_acc | I32 | [8,10] | 320 | 10 | 11 | no |
| Logits | F32 | [8,10] | 320 | 11 | 12 | no |
| Probs (output) | F32 | [8,10] | 320 | 0 | 13 | yes |
| Preds (output) | I32 | [8] | 32 | 0 | 13 | yes |

`H1`'s lifetime `[3, 8]` spans the *entire second layer* (nodes 4–7: quantize, matmul, dequantize, relu) — it's produced right after the first ReLU and not consumed until the residual `Add` five nodes later. Meanwhile `Xq`, `H1_acc`, and `H1_pre` (all already dead by node 3/4) are legitimate candidates for reuse by `H1q`, `H2_acc`, `H2_pre`.

Running the algorithm above (measured directly, via `Engine.arena_size_bytes` — see `tests/test_arena_planner.py` and `tests/test_end_to_end.py` for the regression-pinned assertions):

- **Naive sum** of all 15 tensors above: **63,072 bytes**.
- **Planned arena size: 39,904 bytes** — a ~37% reduction, achieved while `H1`'s 4,096-byte block sits untouched through the entire second layer's computation, and the pinned `X`/`Probs`/`Preds` tensors (25,440 bytes combined) are never touched by anything else.

If you change the topology or the allocator and this number moves, that's expected — update the regression-pinned constants in `tests/test_arena_planner.py` and `tests/test_end_to_end.py` (and this document) to match the newly-measured value, rather than loosening the assertions.

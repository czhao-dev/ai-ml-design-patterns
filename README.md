# Tensor Graph Inference Engine

> A minimal, header-only, CPU-only static-graph neural network inference engine (think: a stripped-down ONNX Runtime or GGML). An offline compiler parses a model's DAG, quantizes weights to INT8, and pre-plans a single contiguous memory arena via tensor-lifetime analysis. The header-only runtime then executes a forward pass with **zero dynamic memory allocation**.

## Why

In high-performance/embedded/real-time deployment, calling `new` or `malloc` during inference is a fatal error: it introduces unpredictable latency, can fail under memory pressure, and defeats static analysis of worst-case execution time. The standard fix is to move all memory planning **offline**: parse the network's computation graph once, at compile time, compute the exact lifetime of every intermediate tensor, and pack them into a single pre-sized arena, reusing space for tensors whose lifetimes don't overlap. The runtime then just loads that plan and executes with pure pointer arithmetic.

This project builds that pipeline end to end for a small residual MLP:

- an **offline compiler** (`tools/compile_model.cpp`) that builds an in-memory DAG, quantizes weights/activations to INT8 (per-tensor symmetric scale, INT32 accumulation, fused bias), runs a greedy interval-allocation memory planner, and serializes everything into a flat binary artifact;
- a **header-only runtime** (`include/tge/*.hpp`) that loads that artifact and runs `forward()` touching only pre-planned memory — proven with a test that overrides `operator new`/`delete` and asserts zero allocations during inference.

## What It Builds

The demo network is a 3-layer MLP **with a residual/skip connection** (not just a linear chain) so the memory planner has to prove something non-trivial: keep one activation tensor alive across several unrelated intermediate computations while reclaiming other buffers underneath it.

```text
Input (fp32, [8, 784])
  -> quantize -> INT8 matmul + INT32 bias (784x128) -> dequantize -> ReLU  =: H1  ---------.
                                                                                            |
  H1 -> quantize -> INT8 matmul + INT32 bias (128x128) -> dequantize -> ReLU -> Add <-------'
                                                                                  |
                                                     quantize -> INT8 matmul + INT32 bias (128x10)
                                                                                  |
                                                                    dequantize -> softmax -> argmax
```

`H1` (produced after the first ReLU) is read again by the `Add` node five ops later — it must stay resident through the second layer's quantize/matmul/dequantize/relu, while the buffers used *by* those ops (already dead by then) get reused elsewhere in the arena.

## Repository Layout

```text
.
├── CMakeLists.txt / Makefile      # CXX-only build, CPU-only
├── include/tge/                   # the header-only library
│   ├── types.hpp                  # DType/TensorKind/OpType, alignment helpers
│   ├── ops.hpp                    # quantize/dequantize, INT8 matmul, relu, add, softmax, argmax
│   ├── fp32_reference.hpp         # fp32 ground truth + synthetic data generators
│   ├── graph.hpp                  # offline in-memory graph + GraphBuilder
│   ├── arena_planner.hpp          # tensor lifetime analysis + greedy arena allocator
│   ├── model_format.hpp           # byte-precise compiled artifact (read + write)
│   ├── engine.hpp                 # the zero-allocation runtime
│   └── demo_graph.hpp             # the concrete demo topology (shared by tool + tests)
├── tools/compile_model.cpp        # offline compiler CLI
├── src/infer_main.cpp             # CLI runner
├── tests/                         # op-level, arena-planner, end-to-end, zero-alloc tests
└── docs/arena_design.md           # full pseudocode + worked example for the planner
```

## Quantization Scheme

Per-tensor **symmetric** INT8 quantization, zero-point fixed at `0`:

```
scale = max(abs(tensor)) / 127
quantized = clamp(round(value / scale), -128, 127)
```

Weights and biases are quantized once, offline, by the compiler. Activation scales are calibrated from a single forward pass over representative input at compile time (matching the old GPU-kernel project's "compute hidden calibration" step). Each INT8 matmul accumulates in INT32 with the (already-quantized) bias fused directly into the accumulator — the "scale-shifting" step is the subsequent dequantize, `fp32_value = int32_accumulator * (input_scale * weight_scale)`.

The INT8 matmul itself (`op_matmul_int8` in `include/tge/ops.hpp`) is a K-blocked scalar GEMM: this is the CPU-idiomatic analogue of a GPU shared-memory-tiled kernel, not a literal port — block-level thread cooperation and barriers have no CPU equivalent. Blocking over K keeps the accumulator row and a K-slice of the weight matrix cache-resident, and the inner loop over the output dimension is unit-stride so the compiler can auto-vectorize it (the scalar analogue of DP4A/VNNI-style INT8 dot-product grouping on GPU tensor cores).

## Static Memory Arena

See [`docs/arena_design.md`](docs/arena_design.md) for the full algorithm and a worked example. In short:

1. **`compute_lifetimes()`** walks the topologically-sorted node list once and records, for every intermediate tensor, `[start_node, end_node]` — the node that produces it and the last node that consumes it. Graph inputs/outputs are pinned for the whole run.
2. **`plan_arena()`** sorts tensors by start time and runs a greedy first-fit interval allocator: as each tensor's lifetime begins, any earlier (non-pinned) tensor whose lifetime has already ended is returned to a free list and its space reused.
3. The result — one offset per tensor and a total arena size — is baked into the compiled artifact. The runtime allocates that many bytes **once**, at load time, and never again.

For the demo model at batch size 8, the naive "sum of every intermediate tensor" would need **63,072 bytes**; the planner produces an arena of **39,904 bytes** — a ~37% reduction purely from lifetime-aware reuse, with the residual tensor `H1`'s slot provably untouched by anything allocated during the second layer's computation.

## Compiled Artifact Format (`.tge`)

A single flat binary file, back to back with no gaps:

```
FileHeader (64B)  ->  TensorDesc[num_tensors] (48B each)  ->  NodeDesc[num_nodes] (24B each)  ->  weights_blob
```

The arena's *bytes* are never stored on disk, only its *size* — the runtime allocates a fresh zero-initialized buffer at load time and computes every tensor's address as `arena_base + tensor_table[id].offset`. See `include/tge/model_format.hpp` for the exact struct layouts.

## Build

```bash
cmake -S . -B build/cmake
cmake --build build/cmake
ctest --test-dir build/cmake --output-on-failure
```

Or with the Makefile:

```bash
make all
make test
```

## Run

```bash
./build/cmake/compile_model --output models/demo.tge --batch 8
./build/cmake/infer --model models/demo.tge
```

`compile_model` prints tensor/node counts, arena size, and weights-blob size. `infer` loads the artifact, runs a forward pass on synthetic input, and prints per-row predictions and probabilities.

## Testing

- **`test_ops`** — quantize/dequantize round-trip, INT8 matmul vs. an fp32 reference within a derived quantization-error bound, relu, softmax, argmax.
- **`test_arena_planner`** — synthetic DAGs with known overlapping/disjoint lifetimes, asserting exact offsets (disjoint lifetimes reuse the same offset, overlapping lifetimes never alias, pinned tensors are never reused), plus a regression pin of the real demo graph's arena size.
- **`test_end_to_end`** — the full INT8 demo graph vs. the fp32 reference forward pass over the same weights/input, bounded probability error, prediction agreement.
- **`test_zero_alloc`** — overrides global `operator new`/`delete` with a counting shim and asserts `forward()`/`set_input()` allocate exactly zero bytes across repeated calls; allocation only happens once, during `Engine`'s constructor (model loading).

## Possible Extensions

- Real SIMD intrinsics (AVX2/AVX-512 VNNI `_mm256_dpbusd_epi32` or ARM NEON `vdotq`) for the INT8 GEMM instead of relying on auto-vectorization.
- Per-channel (rather than per-tensor) weight quantization for better accuracy at low bit-width.
- A general topological sort in `GraphBuilder`, rather than relying on construction order (fine for a hand-built demo graph; would matter for a graph assembled from an arbitrary op list).
- Best-fit or slab-class allocation strategies in the arena planner instead of first-fit, to reduce fragmentation on larger/more irregular graphs.

## Further Reading

- Jacob, B., et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." *CVPR*, 2018. [arxiv.org/abs/1712.05877](https://arxiv.org/abs/1712.05877)
- [Arena design notes](docs/arena_design.md)

// Arena planner correctness tests: synthetic DAGs with known
// overlapping/disjoint lifetimes, asserting exact offsets/sizes, plus a
// regression pin against the real demo graph.

#include <cassert>
#include <cstdio>
#include <vector>

#include "tge/arena_planner.hpp"
#include "tge/demo_graph.hpp"
#include "tge/graph.hpp"
#include "tge/types.hpp"

using namespace tge;

// Two tensors with strictly disjoint lifetimes and equal size must be
// assigned the SAME offset (proves reuse happened).
static void test_disjoint_lifetimes_reuse_same_offset() {
    std::vector<TensorLifetime> lifetimes = {
        {/*tensor_id=*/0, /*start=*/0, /*end=*/1, /*size=*/64, /*pinned=*/false},
        {/*tensor_id=*/1, /*start=*/2, /*end=*/3, /*size=*/64, /*pinned=*/false},
    };
    ArenaPlan plan = plan_arena(lifetimes, 16);
    assert(plan.assignments.size() == 2);
    assert(plan.assignments[0].offset == plan.assignments[1].offset);
    assert(plan.total_size_bytes == 64);
    std::printf("test_disjoint_lifetimes_reuse_same_offset: OK\n");
}

// Two tensors with overlapping lifetimes must get non-intersecting byte
// ranges.
static void test_overlapping_lifetimes_disjoint_offsets() {
    std::vector<TensorLifetime> lifetimes = {
        {0, 0, 5, 64, false},
        {1, 2, 3, 128, false},
    };
    ArenaPlan plan = plan_arena(lifetimes, 16);
    uint32_t off0 = 0, off1 = 0;
    for (const auto& a : plan.assignments) {
        if (a.tensor_id == 0) off0 = a.offset;
        if (a.tensor_id == 1) off1 = a.offset;
    }
    bool disjoint = (off0 + 64 <= off1) || (off1 + 128 <= off0);
    assert(disjoint);
    assert(plan.total_size_bytes == 64 + 128);
    std::printf("test_overlapping_lifetimes_disjoint_offsets: OK\n");
}

// A pinned tensor is never reused and spans the whole graph; its
// (forced) end_node must not cause it to be offered to the free list.
static void test_pinned_tensor_never_reused() {
    std::vector<TensorLifetime> lifetimes = {
        {0, 0, 9, 64, true},   // pinned input, spans entire graph
        {1, 0, 1, 64, false},  // dies early
        {2, 2, 9, 64, false},  // could reuse tensor 1's slot, must not touch tensor 0's
    };
    ArenaPlan plan = plan_arena(lifetimes, 16);
    uint32_t off0 = 0, off1 = 0, off2 = 0;
    for (const auto& a : plan.assignments) {
        if (a.tensor_id == 0) off0 = a.offset;
        if (a.tensor_id == 1) off1 = a.offset;
        if (a.tensor_id == 2) off2 = a.offset;
    }
    assert(off1 == off2);       // tensor 1's slot got reused by tensor 2
    assert(off0 != off1);       // pinned tensor 0 never touched
    std::printf("test_pinned_tensor_never_reused: OK\n");
}

// Hand-constructed case with known round-number sizes: assert the EXACT
// expected total, not just "less than the naive sum".
static void test_exact_total_with_known_overlap() {
    // Lifetimes:  A[0,1] sz64   B[1,2] sz64   C[2,3] sz64 (reuses A's slot)
    // A and B overlap at node 1 (A end=1, B start=1 -> not expired, "1<1" false).
    // B and C overlap at node 2 similarly, so C must reuse A's slot instead.
    std::vector<TensorLifetime> lifetimes = {
        {0, 0, 1, 64, false},
        {1, 1, 2, 64, false},
        {2, 2, 3, 64, false},
    };
    ArenaPlan plan = plan_arena(lifetimes, 16);
    uint32_t off0 = 0, off1 = 0, off2 = 0;
    for (const auto& a : plan.assignments) {
        if (a.tensor_id == 0) off0 = a.offset;
        if (a.tensor_id == 1) off1 = a.offset;
        if (a.tensor_id == 2) off2 = a.offset;
    }
    assert(off0 != off1);
    assert(off1 != off2);
    assert(off0 == off2);
    assert(plan.total_size_bytes == 128);  // exactly 2 concurrent slots needed, not 3
    std::printf("test_exact_total_with_known_overlap: OK\n");
}

// Regression pin against the real demo graph: measure once, hard-code
// the observed value so future changes to the planner or topology that
// regress the arena size are caught.
static void test_demo_graph_arena_regression() {
    // demo::build() already runs compute_lifetimes/plan_arena internally;
    // inspect its reported arena size.
    tge::demo::BuildResult result = tge::demo::build("/tmp/tge_test_arena_regression.tge", 8, 7);
    std::printf("demo graph arena size: %u bytes (tensors=%u nodes=%u weights_blob=%u)\n",
                result.arena_size_bytes, result.num_tensors, result.num_nodes, result.weights_blob_size_bytes);
    assert(result.num_tensors == 21);
    assert(result.num_nodes == 14);
    // Naive sum of all 15 non-weight tensors at batch=8 is 63,072 bytes.
    // Measured once at implementation time: 39,904 bytes (a ~37% cut from
    // lifetime-aware reuse). Hard-coded so a future change to the planner
    // or topology that regresses the arena size gets caught deliberately.
    assert(result.arena_size_bytes == 39904);
    std::printf("test_demo_graph_arena_regression: OK\n");
}

int main() {
    test_disjoint_lifetimes_reuse_same_offset();
    test_overlapping_lifetimes_disjoint_offsets();
    test_pinned_tensor_never_reused();
    test_exact_total_with_known_overlap();
    test_demo_graph_arena_regression();
    std::printf("All test_arena_planner checks passed.\n");
    return 0;
}

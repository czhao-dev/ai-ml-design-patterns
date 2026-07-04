#pragma once

// Offline static memory planner: the centerpiece of the project.
//
// Given a topologically-sorted node list, compute the exact lifetime of
// every intermediate activation tensor (the range of node indices during
// which it must remain valid), then run a greedy interval/linear-scan
// allocator to assign byte offsets within a single contiguous arena --
// reusing space for tensors whose lifetimes don't overlap.
//
// Weight/Bias tensors are NOT planned here: they're placed by a trivial
// sequential bump-allocator into a separate, never-reused weights blob
// (see model_format.hpp's write_artifact). Only Activation/Input/Output
// tensors participate in arena planning.

#include <algorithm>
#include <cstdint>
#include <vector>

#include "tge/graph.hpp"
#include "tge/types.hpp"

namespace tge {

struct TensorLifetime {
    uint32_t tensor_id;
    int start_node;
    int end_node;
    uint32_t size_bytes;
    bool pinned;
};

struct ArenaAssignment {
    uint32_t tensor_id;
    uint32_t offset;
};

struct ArenaPlan {
    std::vector<ArenaAssignment> assignments;
    uint32_t total_size_bytes;
};

inline std::vector<TensorLifetime> compute_lifetimes(const std::vector<TensorInfo>& tensors,
                                                      const std::vector<NodeInfo>& nodes,
                                                      uint32_t alignment) {
    std::vector<TensorLifetime> result;
    const int last_node = static_cast<int>(nodes.size()) - 1;

    for (const TensorInfo& t : tensors) {
        if (t.kind != TensorKind::Activation && t.kind != TensorKind::Input && t.kind != TensorKind::Output) {
            continue;
        }

        TensorLifetime life;
        life.tensor_id = t.id;

        uint32_t count = 1;
        for (uint32_t d : t.dims) count *= d;
        life.size_bytes = align_up(count * dtype_size(t.dtype), alignment);

        if (t.kind == TensorKind::Input || t.kind == TensorKind::Output) {
            life.start_node = 0;
            life.end_node = last_node;
            life.pinned = true;
        } else {
            life.pinned = false;
            int start = 0;
            for (const NodeInfo& n : nodes) {
                if (n.output == t.id) {
                    start = static_cast<int>(n.index);
                    break;
                }
            }
            int end = start;
            bool consumed = false;
            for (const NodeInfo& n : nodes) {
                for (uint32_t in_id : n.inputs) {
                    if (in_id == t.id) {
                        end = std::max(end, static_cast<int>(n.index));
                        consumed = true;
                    }
                }
            }
            if (!consumed) end = last_node;
            life.start_node = start;
            life.end_node = end;
        }

        result.push_back(life);
    }

    return result;
}

inline ArenaPlan plan_arena(std::vector<TensorLifetime> lifetimes, uint32_t alignment) {
    std::sort(lifetimes.begin(), lifetimes.end(), [](const TensorLifetime& a, const TensorLifetime& b) {
        if (a.start_node != b.start_node) return a.start_node < b.start_node;
        return a.tensor_id < b.tensor_id;
    });

    struct FreeBlock {
        uint32_t offset;
        uint32_t size;
    };
    struct ActiveBlock {
        uint32_t tensor_id;
        int end_node;
        uint32_t offset;
        uint32_t size;
        bool pinned;
    };

    std::vector<FreeBlock> free_list;
    std::vector<ActiveBlock> active;
    uint32_t high_water = 0;

    ArenaPlan plan;
    plan.assignments.reserve(lifetimes.size());

    for (const TensorLifetime& life : lifetimes) {
        // Expire non-pinned active blocks that are provably dead before
        // this tensor's lifetime starts. Strict '<' (not '<=') so a
        // tensor produced/consumed at the same node boundary as another
        // is never aliased mid-computation.
        for (size_t i = 0; i < active.size();) {
            if (!active[i].pinned && active[i].end_node < life.start_node) {
                free_list.push_back({active[i].offset, active[i].size});
                active.erase(active.begin() + static_cast<long>(i));
            } else {
                ++i;
            }
        }

        // First-fit allocation from the free list.
        uint32_t offset;
        bool allocated = false;
        for (size_t i = 0; i < free_list.size(); ++i) {
            if (free_list[i].size >= life.size_bytes) {
                offset = free_list[i].offset;
                uint32_t remainder_size = free_list[i].size - life.size_bytes;
                uint32_t remainder_offset = free_list[i].offset + life.size_bytes;
                free_list.erase(free_list.begin() + static_cast<long>(i));
                if (remainder_size > 0) {
                    free_list.push_back({remainder_offset, remainder_size});
                }
                allocated = true;
                break;
            }
        }
        if (!allocated) {
            offset = align_up(high_water, alignment);
            high_water = offset + life.size_bytes;
        }

        plan.assignments.push_back({life.tensor_id, offset});
        active.push_back({life.tensor_id, life.end_node, offset, life.size_bytes, life.pinned});
    }

    plan.total_size_bytes = align_up(high_water, alignment);
    return plan;
}

}  // namespace tge

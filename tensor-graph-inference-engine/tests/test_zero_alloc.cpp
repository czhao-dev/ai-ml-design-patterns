// Proves the project's central claim: Engine::forward() and set_input()
// perform zero heap allocation. Overrides global operator new/delete with
// a counting shim and asserts the allocation counter is unchanged across
// repeated forward() calls -- allocation may only happen while
// constructing the Engine (i.e. during load()).

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>

namespace {
size_t g_alloc_count = 0;
bool g_counting_enabled = false;
}  // namespace

void* operator new(size_t size) {
    if (g_counting_enabled) ++g_alloc_count;
    void* ptr = std::malloc(size == 0 ? 1 : size);
    if (ptr == nullptr) throw std::bad_alloc();
    return ptr;
}

void operator delete(void* ptr) noexcept { std::free(ptr); }
void operator delete(void* ptr, size_t) noexcept { std::free(ptr); }

#include "tge/demo_graph.hpp"
#include "tge/engine.hpp"
#include "tge/fp32_reference.hpp"

int main() {
    const std::string path = "/tmp/tge_test_zero_alloc.tge";
    tge::demo::BuildResult built = tge::demo::build(path, /*batch=*/8, /*seed=*/7);

    // Construction (load()) is allowed to allocate -- this is one-time
    // model-loading cost, not part of the "inference" hot path.
    g_counting_enabled = true;
    size_t before_construction = g_alloc_count;
    tge::Engine engine(path);
    size_t after_construction = g_alloc_count;
    assert(after_construction > before_construction);
    std::printf("allocations during Engine construction: %zu\n", after_construction - before_construction);

    // forward()/set_input() must allocate exactly zero bytes, repeatedly.
    for (int iter = 0; iter < 5; ++iter) {
        size_t before = g_alloc_count;
        engine.set_input(built.calibration_input.data(), built.calibration_input.size());
        engine.forward();
        size_t after = g_alloc_count;
        assert(after == before);
    }

    g_counting_enabled = false;
    std::printf("All test_zero_alloc checks passed: forward()/set_input() allocate 0 bytes.\n");
    return 0;
}

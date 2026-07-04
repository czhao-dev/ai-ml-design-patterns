// Offline model compiler: builds the demo graph, quantizes weights,
// plans the static memory arena, and writes the compiled .tge artifact
// that the header-only runtime (tge::Engine) reads.

#include <cstdio>
#include <cstdlib>
#include <string>

#include "tge/demo_graph.hpp"

int main(int argc, char** argv) {
    std::string output_path = "models/demo.tge";
    int batch = 8;
    uint32_t seed = 7;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--batch" && i + 1 < argc) {
            batch = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<uint32_t>(std::atoi(argv[++i]));
        }
    }

    tge::demo::BuildResult result = tge::demo::build(output_path, batch, seed);

    std::printf("Compiled demo model -> %s\n", result.artifact_path.c_str());
    std::printf("  tensors: %u\n", result.num_tensors);
    std::printf("  nodes: %u\n", result.num_nodes);
    std::printf("  arena size: %u bytes\n", result.arena_size_bytes);
    std::printf("  weights blob size: %u bytes\n", result.weights_blob_size_bytes);
    std::printf("  batch: %d\n", batch);
    return 0;
}

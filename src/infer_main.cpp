// CLI runner: loads a compiled .tge artifact via tge::Engine, runs a
// forward pass on synthetic input, and prints predictions.

#include <cstdio>
#include <string>
#include <vector>

#include "tge/engine.hpp"
#include "tge/fp32_reference.hpp"

int main(int argc, char** argv) {
    std::string model_path = "models/demo.tge";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        }
    }

    tge::Engine engine(model_path);
    std::printf("Loaded %s (arena: %u bytes, batch: %u, input: %u, classes: %u)\n", model_path.c_str(),
                engine.arena_size_bytes(), engine.batch(), engine.input_size(), engine.output_classes());

    std::vector<float> input = tge::ref::make_synthetic_input(static_cast<int>(engine.batch()),
                                                               static_cast<int>(engine.input_size()), 11);
    engine.set_input(input.data(), input.size());
    engine.forward();

    const float* probs = engine.probabilities();
    const int32_t* preds = engine.predictions();
    const uint32_t classes = engine.output_classes();

    for (uint32_t row = 0; row < engine.batch(); ++row) {
        std::printf("row %u: prediction=%d probs=[", row, preds[row]);
        for (uint32_t c = 0; c < classes; ++c) {
            std::printf("%s%.4f", c == 0 ? "" : " ", probs[row * classes + c]);
        }
        std::printf("]\n");
    }
    return 0;
}

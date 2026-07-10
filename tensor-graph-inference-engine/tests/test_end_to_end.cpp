// End-to-end test: compiles the demo graph, runs it through tge::Engine,
// and compares against the fp32 reference forward pass computed over the
// exact same weights/input. Also pins the measured arena size as a
// regression value.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "tge/demo_graph.hpp"
#include "tge/engine.hpp"
#include "tge/fp32_reference.hpp"

using namespace tge;

int main() {
    const std::string path = "/tmp/tge_test_end_to_end.tge";
    demo::BuildResult built = demo::build(path, /*batch=*/8, /*seed=*/7);

    Engine engine(path);
    assert(engine.batch() == 8);
    assert(engine.input_size() == demo::kInputDim);
    assert(engine.output_classes() == demo::kOutputDim);

    engine.set_input(built.calibration_input.data(), built.calibration_input.size());
    engine.forward();

    const float* probs = engine.probabilities();
    const int32_t* preds = engine.predictions();
    const int batch = static_cast<int>(engine.batch());
    const int classes = static_cast<int>(engine.output_classes());

    // Probabilities are a valid distribution.
    for (int r = 0; r < batch; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < classes; ++c) {
            float p = probs[r * classes + c];
            assert(p >= 0.0f && p <= 1.0f);
            sum += p;
        }
        assert(std::fabs(sum - 1.0f) < 1e-3f);
    }

    // Bounded error against the fp32 reference computed over the same
    // weights/input (INT8 quantization introduces error, but it must stay
    // small and predictions should usually agree).
    std::vector<float> quant_probs(probs, probs + static_cast<size_t>(batch) * classes);
    float max_err = ref::max_abs_error(quant_probs, built.fp32_reference.probabilities);
    float mean_err = ref::mean_abs_error(quant_probs, built.fp32_reference.probabilities);
    std::printf("probability error vs fp32 reference: max=%.6f mean=%.6f\n", max_err, mean_err);
    assert(max_err < 0.25f);

    int agree = 0;
    for (int r = 0; r < batch; ++r) {
        if (preds[r] == built.fp32_reference.predictions[r]) ++agree;
    }
    std::printf("predictions agreeing with fp32 reference: %d/%d\n", agree, batch);
    assert(agree >= batch / 2);

    // Regression pin: measured once at implementation time (naive sum of
    // all 15 non-weight tensors at batch=8 is 63,072 bytes; the planner
    // achieves 39,904 via lifetime-aware reuse). If this ever fails,
    // either the topology or the planner changed -- both worth a
    // deliberate look, not a silent drift.
    std::printf("measured arena_size_bytes=%u\n", engine.arena_size_bytes());
    assert(engine.arena_size_bytes() == built.arena_size_bytes);
    assert(built.arena_size_bytes == 39904);

    std::printf("All test_end_to_end checks passed.\n");
    return 0;
}

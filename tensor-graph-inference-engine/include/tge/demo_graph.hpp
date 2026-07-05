#pragma once

// Concrete demo topology: a 3-layer MLP with a residual/skip connection.
//
//   0:  Quantize(X)              -> Xq
//   1:  MatmulInt8(Xq,W1,B1)     -> H1_acc
//   2:  Dequantize(H1_acc)       -> H1_pre
//   3:  Relu(H1_pre)             -> H1          <- residual source, lifetime [3,8]
//   4:  Quantize(H1)             -> H1q
//   5:  MatmulInt8(H1q,W2,B2)    -> H2_acc
//   6:  Dequantize(H2_acc)       -> H2_pre
//   7:  Relu(H2_pre)             -> H2_relu
//   8:  Add(H2_relu,H1)          -> H2_sum       <- consumes H1 (dies here)
//   9:  Quantize(H2_sum)         -> H2q
//   10: MatmulInt8(H2q,W3,B3)    -> Logits_acc
//   11: Dequantize(Logits_acc)   -> Logits
//   12: Softmax(Logits)          -> Probs
//   13: Argmax(Probs)            -> Preds
//
// H1 must stay resident through nodes 4-7 while Xq/H1_acc/H1_pre (already
// dead) get reclaimed for H1q/H2_acc/H2_pre -- this is what actually
// exercises lifetime-aware arena reuse; a linear chain wouldn't.
//
// Shared by tools/compile_model.cpp and tests, via demo::build(), so the
// compiled artifact and the fp32 ground truth can never drift apart.

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tge/arena_planner.hpp"
#include "tge/fp32_reference.hpp"
#include "tge/graph.hpp"
#include "tge/model_format.hpp"
#include "tge/ops.hpp"
#include "tge/types.hpp"

namespace tge::demo {

constexpr int kInputDim = 784;
constexpr int kHiddenDim = 128;
constexpr int kOutputDim = 10;

struct BuildResult {
    std::string artifact_path;
    ref::DemoWeights weights;
    std::vector<float> calibration_input;
    ref::DemoForwardResult fp32_reference;
    uint32_t num_tensors = 0;
    uint32_t num_nodes = 0;
    uint32_t arena_size_bytes = 0;
    uint32_t weights_blob_size_bytes = 0;
};

inline std::vector<int8_t> quantize_weight(const std::vector<float>& values, float scale) {
    std::vector<int8_t> out(values.size());
    op_quantize_f32_to_i8(values.data(), out.data(), values.size(), scale);
    return out;
}

inline std::vector<int32_t> quantize_bias(const std::vector<float>& values, float combined_scale) {
    std::vector<int32_t> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = static_cast<int32_t>(std::lround(values[i] / combined_scale));
    }
    return out;
}

inline BuildResult build(const std::string& output_path, int batch = 8, uint32_t seed = 7) {
    BuildResult result;
    result.artifact_path = output_path;
    result.weights = ref::make_synthetic_demo_weights(kInputDim, kHiddenDim, kOutputDim, seed);
    result.calibration_input = ref::make_synthetic_input(batch, kInputDim, seed + 4);
    result.fp32_reference =
        ref::run_fp32_reference(result.weights, result.calibration_input, batch, kInputDim, kHiddenDim, kOutputDim);

    // Calibration: one-shot pass over the synthetic input to derive
    // per-tensor symmetric scales for every activation and weight.
    const float input_scale = compute_symmetric_scale(result.calibration_input.data(), result.calibration_input.size());
    const float w1_scale = compute_symmetric_scale(result.weights.w1.data(), result.weights.w1.size());
    const float w2_scale = compute_symmetric_scale(result.weights.w2.data(), result.weights.w2.size());
    const float w3_scale = compute_symmetric_scale(result.weights.w3.data(), result.weights.w3.size());
    const float hidden1_scale = compute_symmetric_scale(result.fp32_reference.h1.data(), result.fp32_reference.h1.size());
    const float hidden2_scale =
        compute_symmetric_scale(result.fp32_reference.h2_sum.data(), result.fp32_reference.h2_sum.size());

    const float combined1_scale = input_scale * w1_scale;
    const float combined2_scale = hidden1_scale * w2_scale;
    const float combined3_scale = hidden2_scale * w3_scale;

    GraphBuilder g;
    uint32_t x = g.add_input("X", {static_cast<uint32_t>(batch), kInputDim});
    uint32_t w1 = g.add_weight("W1", {kInputDim, kHiddenDim}, quantize_weight(result.weights.w1, w1_scale), w1_scale);
    uint32_t b1 = g.add_bias("B1", {kHiddenDim}, quantize_bias(result.weights.b1, combined1_scale));
    uint32_t w2 = g.add_weight("W2", {kHiddenDim, kHiddenDim}, quantize_weight(result.weights.w2, w2_scale), w2_scale);
    uint32_t b2 = g.add_bias("B2", {kHiddenDim}, quantize_bias(result.weights.b2, combined2_scale));
    uint32_t w3 = g.add_weight("W3", {kHiddenDim, kOutputDim}, quantize_weight(result.weights.w3, w3_scale), w3_scale);
    uint32_t b3 = g.add_bias("B3", {kOutputDim}, quantize_bias(result.weights.b3, combined3_scale));

    uint32_t xq = g.add_quantize(x, input_scale, "Xq");
    uint32_t h1_acc = g.add_matmul_int8(xq, w1, b1, "H1_acc");
    uint32_t h1_pre = g.add_dequantize(h1_acc, combined1_scale, "H1_pre");
    uint32_t h1 = g.add_relu(h1_pre, "H1");

    uint32_t h1q = g.add_quantize(h1, hidden1_scale, "H1q");
    uint32_t h2_acc = g.add_matmul_int8(h1q, w2, b2, "H2_acc");
    uint32_t h2_pre = g.add_dequantize(h2_acc, combined2_scale, "H2_pre");
    uint32_t h2_relu = g.add_relu(h2_pre, "H2_relu");
    uint32_t h2_sum = g.add_add(h2_relu, h1, "H2_sum");

    uint32_t h2q = g.add_quantize(h2_sum, hidden2_scale, "H2q");
    uint32_t logits_acc = g.add_matmul_int8(h2q, w3, b3, "Logits_acc");
    uint32_t logits = g.add_dequantize(logits_acc, combined3_scale, "Logits");
    uint32_t probs = g.add_softmax(logits, "Probs");
    uint32_t preds = g.add_argmax(probs, "Preds");

    g.mark_output(probs);
    g.mark_output(preds);

    auto lifetimes = compute_lifetimes(g.tensors(), g.nodes(), kAlignment);
    auto arena_plan = plan_arena(std::move(lifetimes), kAlignment);

    if (!write_artifact(output_path, g.tensors(), g.nodes(), arena_plan, kAlignment)) {
        throw std::runtime_error("tge: failed to write artifact to " + output_path);
    }

    result.num_tensors = static_cast<uint32_t>(g.tensors().size());
    result.num_nodes = static_cast<uint32_t>(g.nodes().size());
    result.arena_size_bytes = arena_plan.total_size_bytes;
    uint32_t weights_bytes = 0;
    for (const TensorInfo& t : g.tensors()) {
        if (t.kind == TensorKind::Weight || t.kind == TensorKind::Bias) {
            weights_bytes = align_up(weights_bytes, kAlignment) + static_cast<uint32_t>(t.static_data.size());
        }
    }
    result.weights_blob_size_bytes = weights_bytes;
    return result;
}

}  // namespace tge::demo

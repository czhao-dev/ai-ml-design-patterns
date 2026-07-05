// Op-level unit tests: quantize/dequantize round-trip, int8 matmul vs a
// naive fp32 reference within quantization error bounds, relu, softmax,
// argmax. No dependency on the graph/engine machinery.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

#include "tge/fp32_reference.hpp"
#include "tge/ops.hpp"

using namespace tge;

static void test_quantize_dequantize_roundtrip() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> values(256);
    for (auto& v : values) v = dist(rng);

    float scale = compute_symmetric_scale(values.data(), values.size());
    std::vector<int8_t> quantized(values.size());
    op_quantize_f32_to_i8(values.data(), quantized.data(), values.size(), scale);

    std::vector<int32_t> as_i32(values.size());
    for (size_t i = 0; i < values.size(); ++i) as_i32[i] = quantized[i];
    std::vector<float> dequantized(values.size());
    op_dequantize_i32_to_f32(as_i32.data(), dequantized.data(), values.size(), scale);

    for (size_t i = 0; i < values.size(); ++i) {
        assert(std::fabs(values[i] - dequantized[i]) <= scale + 1e-6f);
    }
    std::printf("test_quantize_dequantize_roundtrip: OK\n");
}

static void test_matmul_int8_vs_fp32_reference() {
    const int m = 5, k = 37, n = 11;  // deliberately not multiples of any block size
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> act_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> weight_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> bias_dist(-0.1f, 0.1f);

    std::vector<float> act_f(static_cast<size_t>(m) * k);
    std::vector<float> weight_f(static_cast<size_t>(k) * n);
    std::vector<float> bias_f(n);
    for (auto& v : act_f) v = act_dist(rng);
    for (auto& v : weight_f) v = weight_dist(rng);
    for (auto& v : bias_f) v = bias_dist(rng);

    std::vector<float> fp32_out(static_cast<size_t>(m) * n);
    ref::dense(act_f.data(), weight_f.data(), bias_f.data(), fp32_out.data(), m, n, k);

    float act_scale = compute_symmetric_scale(act_f.data(), act_f.size());
    float weight_scale = compute_symmetric_scale(weight_f.data(), weight_f.size());
    float combined_scale = act_scale * weight_scale;

    std::vector<int8_t> act_q(act_f.size());
    op_quantize_f32_to_i8(act_f.data(), act_q.data(), act_f.size(), act_scale);
    std::vector<int8_t> weight_q(weight_f.size());
    op_quantize_f32_to_i8(weight_f.data(), weight_q.data(), weight_f.size(), weight_scale);
    std::vector<int32_t> bias_q(n);
    for (int j = 0; j < n; ++j) bias_q[j] = static_cast<int32_t>(std::lround(bias_f[j] / combined_scale));

    std::vector<int32_t> acc(static_cast<size_t>(m) * n);
    op_matmul_int8(act_q.data(), weight_q.data(), bias_q.data(), acc.data(), m, n, k);
    std::vector<float> quant_out(acc.size());
    op_dequantize_i32_to_f32(acc.data(), quant_out.data(), acc.size(), combined_scale);

    // Bounded quantization error. Per k-term, the dominant error source is
    // rounding one operand while the other stays at near-full magnitude:
    // |a_true|*dw + |w_true|*da, bounded by max|a|*weight_scale +
    // max|w|*act_scale. Errors across k terms have effectively random
    // sign, so they accumulate like a random walk (~sqrt(k)), not linearly
    // -- hence the sqrt(k) scaling with a generous safety factor below.
    float max_act = 0.0f, max_weight = 0.0f;
    for (float v : act_f) max_act = std::max(max_act, std::fabs(v));
    for (float v : weight_f) max_weight = std::max(max_weight, std::fabs(v));
    float per_term_bound = max_act * weight_scale + max_weight * act_scale;
    float tolerance = per_term_bound * std::sqrt(static_cast<float>(k)) * 2.0f + combined_scale;

    float max_err = ref::max_abs_error(fp32_out, quant_out);
    assert(max_err <= tolerance);
    std::printf("test_matmul_int8_vs_fp32_reference: OK (max_err=%.6f tolerance=%.6f)\n", max_err, tolerance);
}

static void test_relu() {
    std::vector<float> in = {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f};
    std::vector<float> out(in.size());
    op_relu_f32(in.data(), out.data(), in.size());
    std::vector<float> expected = {0.0f, 0.0f, 0.0f, 0.5f, 2.0f};
    for (size_t i = 0; i < in.size(); ++i) assert(out[i] == expected[i]);
    std::printf("test_relu: OK\n");
}

static void test_softmax_and_argmax() {
    std::vector<float> logits = {1.0f, 2.0f, 0.5f, 3.0f, 2.9f, -1.0f};  // 2 rows x 3 cols
    std::vector<float> probs(logits.size());
    op_softmax_f32(logits.data(), probs.data(), 2, 3);

    for (int r = 0; r < 2; ++r) {
        float sum = probs[r * 3] + probs[r * 3 + 1] + probs[r * 3 + 2];
        assert(std::fabs(sum - 1.0f) < 1e-5f);
        for (int c = 0; c < 3; ++c) assert(probs[r * 3 + c] >= 0.0f && probs[r * 3 + c] <= 1.0f);
    }

    std::vector<int32_t> preds(2);
    op_argmax_i32(probs.data(), preds.data(), 2, 3);
    assert(preds[0] == 1);  // logit 2.0 is max in row 0
    assert(preds[1] == 0);  // logit 3.0 is max in row 1
    std::printf("test_softmax_and_argmax: OK\n");
}

int main() {
    test_quantize_dequantize_roundtrip();
    test_matmul_int8_vs_fp32_reference();
    test_relu();
    test_softmax_and_argmax();
    std::printf("All test_ops checks passed.\n");
    return 0;
}

#pragma once

// FP32 ground-truth forward pass and synthetic data generation. Used
// offline only: to calibrate INT8 quantization scales, and as the
// correctness baseline that the quantized graph is checked against in
// tests/test_end_to_end.cpp. Never included by anything on the hot path.

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace tge::ref {

struct DemoWeights {
    std::vector<float> w1, b1;  // [input_dim, hidden_dim], [hidden_dim]
    std::vector<float> w2, b2;  // [hidden_dim, hidden_dim], [hidden_dim]
    std::vector<float> w3, b3;  // [hidden_dim, output_dim], [output_dim]
};

inline DemoWeights make_synthetic_demo_weights(int input_dim, int hidden_dim, int output_dim,
                                                uint32_t seed = 7) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> weight_dist(0.0f, 0.08f);
    std::uniform_real_distribution<float> bias_dist(-0.02f, 0.02f);

    DemoWeights w;
    w.w1.resize(static_cast<size_t>(input_dim) * hidden_dim);
    w.b1.resize(hidden_dim);
    w.w2.resize(static_cast<size_t>(hidden_dim) * hidden_dim);
    w.b2.resize(hidden_dim);
    w.w3.resize(static_cast<size_t>(hidden_dim) * output_dim);
    w.b3.resize(output_dim);

    for (auto& v : w.w1) v = weight_dist(rng);
    for (auto& v : w.b1) v = bias_dist(rng);
    for (auto& v : w.w2) v = weight_dist(rng);
    for (auto& v : w.b2) v = bias_dist(rng);
    for (auto& v : w.w3) v = weight_dist(rng);
    for (auto& v : w.b3) v = bias_dist(rng);
    return w;
}

inline std::vector<float> make_synthetic_input(int batch, int input_dim, uint32_t seed = 11) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> input_dist(0.0f, 1.0f);
    std::vector<float> input(static_cast<size_t>(batch) * input_dim);
    for (auto& v : input) v = input_dist(rng);
    return input;
}

// Naive triple-loop fp32 dense layer: activation [m,k] * weight [k,n] + bias[n] -> out [m,n].
inline void dense(const float* activation, const float* weight, const float* bias, float* out,
                   int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = bias != nullptr ? bias[j] : 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                acc += activation[static_cast<size_t>(i) * k + kk] * weight[static_cast<size_t>(kk) * n + j];
            }
            out[static_cast<size_t>(i) * n + j] = acc;
        }
    }
}

inline void relu(const float* in, float* out, size_t count) {
    for (size_t i = 0; i < count; ++i) out[i] = std::max(in[i], 0.0f);
}

inline void add(const float* a, const float* b, float* out, size_t count) {
    for (size_t i = 0; i < count; ++i) out[i] = a[i] + b[i];
}

inline void softmax(const float* logits, float* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* row_in = logits + static_cast<size_t>(r) * cols;
        float* row_out = out + static_cast<size_t>(r) * cols;
        float row_max = row_in[0];
        for (int c = 1; c < cols; ++c) row_max = std::max(row_max, row_in[c]);
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float e = std::exp(row_in[c] - row_max);
            row_out[c] = e;
            sum += e;
        }
        for (int c = 0; c < cols; ++c) row_out[c] /= sum;
    }
}

inline void argmax(const float* probs, int* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* row = probs + static_cast<size_t>(r) * cols;
        int best = 0;
        for (int c = 1; c < cols; ++c) {
            if (row[c] > row[best]) best = c;
        }
        out[r] = best;
    }
}

struct DemoForwardResult {
    std::vector<float> h1;             // post-ReLU hidden1 (residual source), [batch, hidden_dim]
    std::vector<float> h2_sum;         // post-residual-add, pre-final-relu,   [batch, hidden_dim]
    std::vector<float> logits;         // [batch, output_dim]
    std::vector<float> probabilities;  // [batch, output_dim]
    std::vector<int> predictions;      // [batch]
};

// Ground truth for the demo topology:
//   dense1 -> relu -> dense2 -> relu -> add(residual with h1) -> dense3 -> softmax -> argmax
inline DemoForwardResult run_fp32_reference(const DemoWeights& weights, const std::vector<float>& input,
                                             int batch, int input_dim, int hidden_dim, int output_dim) {
    DemoForwardResult result;
    result.h1.resize(static_cast<size_t>(batch) * hidden_dim);
    result.h2_sum.resize(static_cast<size_t>(batch) * hidden_dim);
    result.logits.resize(static_cast<size_t>(batch) * output_dim);
    result.probabilities.resize(static_cast<size_t>(batch) * output_dim);
    result.predictions.resize(batch);

    std::vector<float> h1_pre(static_cast<size_t>(batch) * hidden_dim);
    dense(input.data(), weights.w1.data(), weights.b1.data(), h1_pre.data(), batch, hidden_dim, input_dim);
    relu(h1_pre.data(), result.h1.data(), h1_pre.size());

    std::vector<float> h2_pre(static_cast<size_t>(batch) * hidden_dim);
    dense(result.h1.data(), weights.w2.data(), weights.b2.data(), h2_pre.data(), batch, hidden_dim, hidden_dim);
    std::vector<float> h2_relu(static_cast<size_t>(batch) * hidden_dim);
    relu(h2_pre.data(), h2_relu.data(), h2_pre.size());
    add(h2_relu.data(), result.h1.data(), result.h2_sum.data(), result.h2_sum.size());

    dense(result.h2_sum.data(), weights.w3.data(), weights.b3.data(), result.logits.data(), batch, output_dim,
          hidden_dim);
    softmax(result.logits.data(), result.probabilities.data(), batch, output_dim);
    argmax(result.probabilities.data(), result.predictions.data(), batch, output_dim);
    return result;
}

inline float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    float max_err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) max_err = std::max(max_err, std::fabs(a[i] - b[i]));
    return max_err;
}

inline float mean_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) sum += std::fabs(a[i] - b[i]);
    return a.empty() ? 0.0f : sum / static_cast<float>(a.size());
}

}  // namespace tge::ref

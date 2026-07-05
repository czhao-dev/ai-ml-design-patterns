#pragma once

// Pure compute kernels for the tensor graph engine. Every function here is
// the only code touched from inside Engine::forward() (see engine.hpp) --
// they operate purely on raw pointers and sizes, with no heap allocation.
//
// Quantization scheme: per-tensor symmetric INT8, zero_point is always 0.
// INT8 x INT8 -> INT32 accumulation, with an optional INT32 bias fused
// directly into the accumulator (the "scale-shifting vector" -- the bias
// vector is the shift applied per output channel before dequantization).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>

namespace tge {

inline void op_quantize_f32_to_i8(const float* in, int8_t* out, size_t count, float scale) {
    const float inv_scale = 1.0f / scale;
    for (size_t i = 0; i < count; ++i) {
        float scaled = std::round(in[i] * inv_scale);
        scaled = std::max(scaled, -128.0f);
        scaled = std::min(scaled, 127.0f);
        out[i] = static_cast<int8_t>(scaled);
    }
}

inline void op_dequantize_i32_to_f32(const int32_t* in, float* out, size_t count, float combined_scale) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<float>(in[i]) * combined_scale;
    }
}

inline void op_relu_f32(const float* in, float* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = std::max(in[i], 0.0f);
    }
}

inline void op_add_f32(const float* a, const float* b, float* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] + b[i];
    }
}

inline void op_softmax_f32(const float* logits, float* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* row_in = logits + static_cast<size_t>(r) * cols;
        float* row_out = out + static_cast<size_t>(r) * cols;

        float row_max = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < cols; ++c) row_max = std::max(row_max, row_in[c]);

        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float e = std::exp(row_in[c] - row_max);
            row_out[c] = e;
            sum += e;
        }

        const float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; ++c) row_out[c] *= inv_sum;
    }
}

inline void op_argmax_i32(const float* probs, int32_t* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* row = probs + static_cast<size_t>(r) * cols;
        int best = 0;
        float best_val = row[0];
        for (int c = 1; c < cols; ++c) {
            if (row[c] > best_val) {
                best_val = row[c];
                best = c;
            }
        }
        out[r] = best;
    }
}

// activation: [m, k] INT8, row-major. weight: [k, n] INT8, row-major.
// bias: [n] INT32, may be nullptr. out: [m, n] INT32, row-major.
//
// K-blocked scalar GEMM: this is the CPU-idiomatic analogue of a GPU
// shared-memory-tiled kernel. It is *not* a literal port -- block-level
// thread cooperation and barriers have no CPU equivalent. Blocking
// over K bounds the working set so the accumulator row and a K-slice of
// the weight matrix stay cache-resident, and the inner loop over `n` is
// unit-stride so the compiler can auto-vectorize it (the scalar analogue
// of DP4A/VNNI-style INT8 dot-product grouping on GPU tensor cores).
inline void op_matmul_int8(const int8_t* activation, const int8_t* weight, const int32_t* bias,
                            int32_t* out, int m, int n, int k) {
    constexpr int kBlockK = 256;

    for (int i = 0; i < m; ++i) {
        int32_t* out_row = out + static_cast<size_t>(i) * n;
        if (bias != nullptr) {
            for (int j = 0; j < n; ++j) out_row[j] = bias[j];
        } else {
            for (int j = 0; j < n; ++j) out_row[j] = 0;
        }
    }

    for (int i = 0; i < m; ++i) {
        const int8_t* act_row = activation + static_cast<size_t>(i) * k;
        int32_t* out_row = out + static_cast<size_t>(i) * n;

        for (int k0 = 0; k0 < k; k0 += kBlockK) {
            const int k1 = std::min(k0 + kBlockK, k);
            for (int kk = k0; kk < k1; ++kk) {
                const int32_t a_val = static_cast<int32_t>(act_row[kk]);
                const int8_t* w_row = weight + static_cast<size_t>(kk) * n;
                for (int j = 0; j < n; ++j) {
                    out_row[j] += a_val * static_cast<int32_t>(w_row[j]);
                }
            }
        }
    }
}

// Offline-only helper (calibration / quantization at compile time, never
// called from forward()): per-tensor symmetric scale.
inline float compute_symmetric_scale(const float* values, size_t count) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < count; ++i) max_abs = std::max(max_abs, std::fabs(values[i]));
    return max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
}

}  // namespace tge

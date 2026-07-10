#pragma once

#include <cstddef>
#include <cstdint>

namespace tge {

enum class DType : uint32_t {
    F32 = 0,
    I8 = 1,
    I32 = 2,
};

enum class TensorKind : uint32_t {
    Activation = 0,
    Weight = 1,
    Bias = 2,
    Input = 3,
    Output = 4,
};

// No `Input` entry here: a graph input is a TensorKind, not an operation --
// it has no producer node.
enum class OpType : uint32_t {
    Quantize = 0,
    MatmulInt8 = 1,
    Dequantize = 2,
    Relu = 3,
    Add = 4,
    Softmax = 5,
    Argmax = 6,
};

constexpr uint32_t kInvalidId = 0xFFFFFFFFu;
constexpr uint32_t kAlignment = 16;
constexpr uint32_t kMaxDims = 4;
constexpr uint32_t kMaxNodeInputs = 3;

inline uint32_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32:
            return 4;
        case DType::I8:
            return 1;
        case DType::I32:
            return 4;
    }
    return 0;
}

inline uint32_t align_up(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// Non-owning view over a tensor's storage. Built on the stack while
// dispatching a node -- never stored, never allocated.
struct TensorView {
    void* data;
    DType dtype;
    uint32_t dims[kMaxDims];
    uint32_t rank;
    float scale;
    int32_t zero_point;

    uint32_t element_count() const {
        uint32_t count = 1;
        for (uint32_t i = 0; i < rank; ++i) count *= dims[i];
        return count;
    }
};

}  // namespace tge

#pragma once

// Offline, in-memory graph representation. Used only by tools/compile_model.cpp
// (via demo_graph.hpp) and tests -- never by the header-only runtime.
//
// Scale-field convention: every TensorInfo::scale is set exactly once, at
// the moment that tensor is created, and always means "the scale needed to
// interpret this tensor's values":
//   - add_weight/add_bias: scale is the caller's precomputed value.
//   - add_quantize(input, scale, name): the *new* I8 output tensor's
//     quantization scale.
//   - add_dequantize(input, combined_scale, name): the *new* F32 output
//     tensor's combined_scale (input_scale * weight_scale), NOT stored on
//     the I32 input. The runtime reads it from the output tensor.
// No tensor's scale is ever mutated after creation.

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "tge/types.hpp"

namespace tge {

struct TensorInfo {
    uint32_t id = kInvalidId;
    std::string name;
    DType dtype = DType::F32;
    TensorKind kind = TensorKind::Activation;
    std::vector<uint32_t> dims;
    float scale = 1.0f;
    int32_t zero_point = 0;
    std::vector<uint8_t> static_data;  // non-empty only for Weight/Bias
};

struct NodeInfo {
    uint32_t index = kInvalidId;
    OpType op;
    std::vector<uint32_t> inputs;  // order matters, see per-op convention below
    uint32_t output = kInvalidId;
};

// Per-op input order convention:
//   Quantize:    [input]
//   MatmulInt8:  [activation, weight, bias]   (bias id may be kInvalidId)
//   Dequantize:  [input]
//   Relu:        [input]
//   Add:         [a, b]
//   Softmax:     [input]
//   Argmax:      [input]
class GraphBuilder {
public:
    uint32_t add_input(std::string name, std::vector<uint32_t> dims, DType dtype = DType::F32) {
        TensorInfo t;
        t.id = next_id_++;
        t.name = std::move(name);
        t.dtype = dtype;
        t.kind = TensorKind::Input;
        t.dims = std::move(dims);
        tensors_.push_back(std::move(t));
        return tensors_.back().id;
    }

    uint32_t add_weight(std::string name, std::vector<uint32_t> dims, std::vector<int8_t> data, float scale) {
        TensorInfo t;
        t.id = next_id_++;
        t.name = std::move(name);
        t.dtype = DType::I8;
        t.kind = TensorKind::Weight;
        t.dims = std::move(dims);
        t.scale = scale;
        t.static_data.resize(data.size());
        std::memcpy(t.static_data.data(), data.data(), data.size());
        tensors_.push_back(std::move(t));
        return tensors_.back().id;
    }

    uint32_t add_bias(std::string name, std::vector<uint32_t> dims, std::vector<int32_t> data) {
        TensorInfo t;
        t.id = next_id_++;
        t.name = std::move(name);
        t.dtype = DType::I32;
        t.kind = TensorKind::Bias;
        t.dims = std::move(dims);
        t.static_data.resize(data.size() * sizeof(int32_t));
        std::memcpy(t.static_data.data(), data.data(), t.static_data.size());
        tensors_.push_back(std::move(t));
        return tensors_.back().id;
    }

    uint32_t add_quantize(uint32_t input, float scale, std::string name) {
        const TensorInfo& in = tensor(input);
        uint32_t out_id = new_activation(std::move(name), in.dims, DType::I8, scale);
        push_node(OpType::Quantize, {input}, out_id);
        return out_id;
    }

    uint32_t add_matmul_int8(uint32_t act, uint32_t weight, uint32_t bias, std::string name) {
        const TensorInfo& a = tensor(act);
        const TensorInfo& w = tensor(weight);
        std::vector<uint32_t> out_dims = {a.dims[0], w.dims[1]};
        uint32_t out_id = new_activation(std::move(name), out_dims, DType::I32, 1.0f);
        push_node(OpType::MatmulInt8, {act, weight, bias}, out_id);
        return out_id;
    }

    uint32_t add_dequantize(uint32_t input, float combined_scale, std::string name) {
        const TensorInfo& in = tensor(input);
        uint32_t out_id = new_activation(std::move(name), in.dims, DType::F32, combined_scale);
        push_node(OpType::Dequantize, {input}, out_id);
        return out_id;
    }

    uint32_t add_relu(uint32_t input, std::string name) {
        const TensorInfo& in = tensor(input);
        uint32_t out_id = new_activation(std::move(name), in.dims, DType::F32, 1.0f);
        push_node(OpType::Relu, {input}, out_id);
        return out_id;
    }

    uint32_t add_add(uint32_t a, uint32_t b, std::string name) {
        const TensorInfo& ta = tensor(a);
        uint32_t out_id = new_activation(std::move(name), ta.dims, DType::F32, 1.0f);
        push_node(OpType::Add, {a, b}, out_id);
        return out_id;
    }

    uint32_t add_softmax(uint32_t input, std::string name) {
        const TensorInfo& in = tensor(input);
        uint32_t out_id = new_activation(std::move(name), in.dims, DType::F32, 1.0f);
        push_node(OpType::Softmax, {input}, out_id);
        return out_id;
    }

    uint32_t add_argmax(uint32_t input, std::string name) {
        const TensorInfo& in = tensor(input);
        std::vector<uint32_t> out_dims = {in.dims[0]};
        uint32_t out_id = new_activation(std::move(name), out_dims, DType::I32, 1.0f);
        push_node(OpType::Argmax, {input}, out_id);
        return out_id;
    }

    void mark_output(uint32_t tensor_id) { tensor_mut(tensor_id).kind = TensorKind::Output; }

    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const std::vector<NodeInfo>& nodes() const { return nodes_; }

private:
    uint32_t new_activation(std::string name, std::vector<uint32_t> dims, DType dtype, float scale) {
        TensorInfo t;
        t.id = next_id_++;
        t.name = std::move(name);
        t.dtype = dtype;
        t.kind = TensorKind::Activation;
        t.dims = std::move(dims);
        t.scale = scale;
        tensors_.push_back(std::move(t));
        return tensors_.back().id;
    }

    void push_node(OpType op, std::vector<uint32_t> inputs, uint32_t output) {
        NodeInfo n;
        n.index = static_cast<uint32_t>(nodes_.size());
        n.op = op;
        n.inputs = std::move(inputs);
        n.output = output;
        nodes_.push_back(std::move(n));
    }

    const TensorInfo& tensor(uint32_t id) const { return tensors_[id]; }
    TensorInfo& tensor_mut(uint32_t id) { return tensors_[id]; }

    std::vector<TensorInfo> tensors_;
    std::vector<NodeInfo> nodes_;
    uint32_t next_id_ = 0;
};

}  // namespace tge

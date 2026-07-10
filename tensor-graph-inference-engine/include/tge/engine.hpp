#pragma once

// The header-only runtime. Engine::Engine() (via load()) is the ONLY place
// this class may allocate: it reads the compiled artifact into a buffer
// (allocation #1) and allocates a zero-initialized arena sized from the
// artifact header (allocation #2). Engine::forward() thereafter does pure
// pointer arithmetic and calls into ops.hpp -- zero heap allocation.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "tge/model_format.hpp"
#include "tge/ops.hpp"
#include "tge/types.hpp"

namespace tge {

class Engine {
public:
    explicit Engine(const std::string& artifact_path) { load(artifact_path); }

    void set_input(const float* data, size_t count) {
        const TensorDesc& desc = view_.tensors[input_id_];
        float* dst = static_cast<float*>(tensor_ptr(desc));
        std::memcpy(dst, data, count * sizeof(float));
    }

    void forward() {
        for (uint32_t i = 0; i < view_.header->num_nodes; ++i) {
            run_node(view_.nodes[i]);
        }
    }

    const float* probabilities() const {
        return static_cast<const float*>(const_cast<Engine*>(this)->tensor_ptr(view_.tensors[probs_id_]));
    }

    const int32_t* predictions() const {
        return static_cast<const int32_t*>(const_cast<Engine*>(this)->tensor_ptr(view_.tensors[preds_id_]));
    }

    uint32_t arena_size_bytes() const { return view_.header->arena_size_bytes; }
    uint32_t batch() const { return view_.tensors[input_id_].dims[0]; }
    uint32_t input_size() const { return view_.tensors[input_id_].dims[1]; }
    uint32_t output_classes() const { return view_.tensors[probs_id_].dims[1]; }

private:
    void load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) throw std::runtime_error("tge: cannot open artifact: " + path);
        std::fseek(f, 0, SEEK_END);
        long size = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (size < 0) {
            std::fclose(f);
            throw std::runtime_error("tge: cannot determine artifact size");
        }

        file_buffer_.resize(static_cast<size_t>(size));  // allocation #1
        size_t read = file_buffer_.empty() ? 0 : std::fread(file_buffer_.data(), 1, file_buffer_.size(), f);
        std::fclose(f);
        if (read != file_buffer_.size()) throw std::runtime_error("tge: failed to read artifact");

        view_ = parse(file_buffer_.data(), file_buffer_.size());
        arena_.assign(view_.header->arena_size_bytes, 0);  // allocation #2

        input_id_ = kInvalidId;
        probs_id_ = kInvalidId;
        preds_id_ = kInvalidId;
        for (uint32_t i = 0; i < view_.header->num_tensors; ++i) {
            const TensorDesc& t = view_.tensors[i];
            if (static_cast<TensorKind>(t.kind) == TensorKind::Input) {
                input_id_ = t.id;
            } else if (static_cast<TensorKind>(t.kind) == TensorKind::Output) {
                // The demo graph marks exactly two outputs: probabilities
                // (F32) and predictions (I32). Distinguish by dtype.
                if (static_cast<DType>(t.dtype) == DType::F32) {
                    probs_id_ = t.id;
                } else {
                    preds_id_ = t.id;
                }
            }
        }
        if (input_id_ == kInvalidId || probs_id_ == kInvalidId || preds_id_ == kInvalidId) {
            throw std::runtime_error("tge: artifact missing required input/output tensors");
        }
    }

    void* tensor_ptr(const TensorDesc& desc) {
        auto kind = static_cast<TensorKind>(desc.kind);
        if (kind == TensorKind::Weight || kind == TensorKind::Bias) {
            return const_cast<uint8_t*>(view_.weights_blob) + desc.offset;
        }
        return arena_.data() + desc.offset;
    }

    void run_node(const NodeDesc& node) {
        auto op = static_cast<OpType>(node.op_type);
        const TensorDesc& out = view_.tensors[node.output_id];

        switch (op) {
            case OpType::Quantize: {
                const TensorDesc& in = view_.tensors[node.input_ids[0]];
                op_quantize_f32_to_i8(static_cast<const float*>(tensor_ptr(in)),
                                       static_cast<int8_t*>(tensor_ptr(out)), out.dims[0] * out.dims[1], out.scale);
                break;
            }
            case OpType::MatmulInt8: {
                const TensorDesc& act = view_.tensors[node.input_ids[0]];
                const TensorDesc& weight = view_.tensors[node.input_ids[1]];
                const int32_t* bias = nullptr;
                if (node.input_ids[2] != kInvalidId) {
                    bias = static_cast<const int32_t*>(tensor_ptr(view_.tensors[node.input_ids[2]]));
                }
                int m = static_cast<int>(act.dims[0]);
                int k = static_cast<int>(act.dims[1]);
                int n = static_cast<int>(weight.dims[1]);
                op_matmul_int8(static_cast<const int8_t*>(tensor_ptr(act)), static_cast<const int8_t*>(tensor_ptr(weight)),
                               bias, static_cast<int32_t*>(tensor_ptr(out)), m, n, k);
                break;
            }
            case OpType::Dequantize: {
                const TensorDesc& in = view_.tensors[node.input_ids[0]];
                op_dequantize_i32_to_f32(static_cast<const int32_t*>(tensor_ptr(in)),
                                          static_cast<float*>(tensor_ptr(out)), out.dims[0] * out.dims[1], out.scale);
                break;
            }
            case OpType::Relu: {
                const TensorDesc& in = view_.tensors[node.input_ids[0]];
                op_relu_f32(static_cast<const float*>(tensor_ptr(in)), static_cast<float*>(tensor_ptr(out)),
                            out.dims[0] * out.dims[1]);
                break;
            }
            case OpType::Add: {
                const TensorDesc& a = view_.tensors[node.input_ids[0]];
                const TensorDesc& b = view_.tensors[node.input_ids[1]];
                op_add_f32(static_cast<const float*>(tensor_ptr(a)), static_cast<const float*>(tensor_ptr(b)),
                           static_cast<float*>(tensor_ptr(out)), out.dims[0] * out.dims[1]);
                break;
            }
            case OpType::Softmax: {
                const TensorDesc& in = view_.tensors[node.input_ids[0]];
                op_softmax_f32(static_cast<const float*>(tensor_ptr(in)), static_cast<float*>(tensor_ptr(out)),
                               static_cast<int>(out.dims[0]), static_cast<int>(out.dims[1]));
                break;
            }
            case OpType::Argmax: {
                const TensorDesc& in = view_.tensors[node.input_ids[0]];
                op_argmax_i32(static_cast<const float*>(tensor_ptr(in)), static_cast<int32_t*>(tensor_ptr(out)),
                              static_cast<int>(in.dims[0]), static_cast<int>(in.dims[1]));
                break;
            }
        }
    }

    std::vector<char> file_buffer_;   // allocation #1 (load)
    std::vector<uint8_t> arena_;      // allocation #2 (load)
    ArtifactView view_{};
    uint32_t input_id_ = kInvalidId;
    uint32_t probs_id_ = kInvalidId;
    uint32_t preds_id_ = kInvalidId;
};

}  // namespace tge

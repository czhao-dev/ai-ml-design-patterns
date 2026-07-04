#pragma once

// Byte-precise compiled model artifact ("TGEM" -- Tensor Graph Engine
// Model). Layout, back to back with no gaps:
//
//   [0                                  ) FileHeader                (64 bytes)
//   [64                                 ) TensorDesc[num_tensors]    (48 bytes each)
//   [64 + 48*num_tensors                ) NodeDesc[num_nodes]        (24 bytes each)
//   [64 + 48*num_tensors + 24*num_nodes ) weights_blob                (weights_blob_size_bytes)
//
// The arena's *bytes* are never stored on disk, only its *size*
// (arena_size_bytes) -- the runtime allocates a fresh zero-initialized
// buffer of that size once at load time (see engine.hpp).
//
// parse() is zero-copy: it validates the header and returns pointers into
// the caller-owned file buffer. write_artifact() is offline-only.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "tge/arena_planner.hpp"
#include "tge/graph.hpp"
#include "tge/types.hpp"

namespace tge {

constexpr char kMagic[4] = {'T', 'G', 'E', 'M'};
constexpr uint32_t kFormatVersion = 1;

struct FileHeader {
    char magic[4];
    uint32_t version;
    uint32_t num_tensors;
    uint32_t num_nodes;
    uint32_t arena_size_bytes;
    uint32_t weights_blob_size_bytes;
    uint32_t alignment;
    uint32_t reserved[9];
};
static_assert(sizeof(FileHeader) == 64, "FileHeader must be exactly 64 bytes");

struct TensorDesc {
    uint32_t id;
    uint32_t dtype;      // DType
    uint32_t kind;       // TensorKind
    uint32_t rank;
    uint32_t dims[kMaxDims];
    uint32_t offset;      // into arena (Activation/Input/Output) or weights_blob (Weight/Bias)
    uint32_t size_bytes;
    float scale;
    int32_t zero_point;
};
static_assert(sizeof(TensorDesc) == 48, "TensorDesc must be exactly 48 bytes");

struct NodeDesc {
    uint32_t op_type;  // OpType
    uint32_t input_ids[kMaxNodeInputs];  // unused slots = kInvalidId
    uint32_t output_id;
    uint32_t reserved;
};
static_assert(sizeof(NodeDesc) == 24, "NodeDesc must be exactly 24 bytes");

struct ArtifactView {
    const FileHeader* header = nullptr;
    const TensorDesc* tensors = nullptr;
    const NodeDesc* nodes = nullptr;
    const uint8_t* weights_blob = nullptr;
};

inline ArtifactView parse(const char* buffer, size_t size) {
    if (size < sizeof(FileHeader)) {
        throw std::runtime_error("tge: artifact too small for header");
    }
    ArtifactView view;
    view.header = reinterpret_cast<const FileHeader*>(buffer);
    if (std::memcmp(view.header->magic, kMagic, 4) != 0) {
        throw std::runtime_error("tge: bad magic in artifact");
    }
    if (view.header->version != kFormatVersion) {
        throw std::runtime_error("tge: unsupported artifact version");
    }

    const size_t tensors_offset = sizeof(FileHeader);
    const size_t nodes_offset = tensors_offset + sizeof(TensorDesc) * view.header->num_tensors;
    const size_t weights_offset = nodes_offset + sizeof(NodeDesc) * view.header->num_nodes;
    const size_t expected_size = weights_offset + view.header->weights_blob_size_bytes;
    if (size < expected_size) {
        throw std::runtime_error("tge: artifact truncated");
    }

    view.tensors = reinterpret_cast<const TensorDesc*>(buffer + tensors_offset);
    view.nodes = reinterpret_cast<const NodeDesc*>(buffer + nodes_offset);
    view.weights_blob = reinterpret_cast<const uint8_t*>(buffer + weights_offset);
    return view;
}

// Offline only: packs Weight/Bias tensors sequentially (with alignment,
// never reused) into a weights blob, then writes the full artifact.
inline bool write_artifact(const std::string& path, const std::vector<TensorInfo>& tensors,
                            const std::vector<NodeInfo>& nodes, const ArenaPlan& arena_plan,
                            uint32_t alignment) {
    std::vector<TensorDesc> tensor_descs(tensors.size());
    std::vector<uint8_t> weights_blob;
    uint32_t weights_high_water = 0;

    for (size_t i = 0; i < tensors.size(); ++i) {
        const TensorInfo& t = tensors[i];
        TensorDesc desc{};
        desc.id = t.id;
        desc.dtype = static_cast<uint32_t>(t.dtype);
        desc.kind = static_cast<uint32_t>(t.kind);
        desc.rank = static_cast<uint32_t>(t.dims.size());
        for (size_t d = 0; d < kMaxDims; ++d) desc.dims[d] = d < t.dims.size() ? t.dims[d] : 1;
        desc.scale = t.scale;
        desc.zero_point = t.zero_point;

        if (t.kind == TensorKind::Weight || t.kind == TensorKind::Bias) {
            desc.size_bytes = static_cast<uint32_t>(t.static_data.size());
            desc.offset = align_up(weights_high_water, alignment);
            weights_blob.resize(desc.offset + desc.size_bytes, 0);
            std::memcpy(weights_blob.data() + desc.offset, t.static_data.data(), t.static_data.size());
            weights_high_water = desc.offset + desc.size_bytes;
        } else {
            desc.size_bytes = 0;
            desc.offset = kInvalidId;
            for (const ArenaAssignment& a : arena_plan.assignments) {
                if (a.tensor_id == t.id) {
                    desc.offset = a.offset;
                    break;
                }
            }
            uint32_t count = 1;
            for (uint32_t dim : t.dims) count *= dim;
            desc.size_bytes = align_up(count * dtype_size(t.dtype), alignment);
        }
        tensor_descs[t.id] = desc;
    }

    std::vector<NodeDesc> node_descs(nodes.size());
    for (const NodeInfo& n : nodes) {
        NodeDesc desc{};
        desc.op_type = static_cast<uint32_t>(n.op);
        desc.input_ids[0] = kInvalidId;
        desc.input_ids[1] = kInvalidId;
        desc.input_ids[2] = kInvalidId;
        for (size_t i = 0; i < n.inputs.size() && i < kMaxNodeInputs; ++i) desc.input_ids[i] = n.inputs[i];
        desc.output_id = n.output;
        desc.reserved = 0;
        node_descs[n.index] = desc;
    }

    FileHeader header{};
    std::memcpy(header.magic, kMagic, 4);
    header.version = kFormatVersion;
    header.num_tensors = static_cast<uint32_t>(tensor_descs.size());
    header.num_nodes = static_cast<uint32_t>(node_descs.size());
    header.arena_size_bytes = arena_plan.total_size_bytes;
    header.weights_blob_size_bytes = static_cast<uint32_t>(weights_blob.size());
    header.alignment = alignment;
    std::memset(header.reserved, 0, sizeof(header.reserved));

    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) return false;
    bool ok = true;
    ok = ok && std::fwrite(&header, sizeof(header), 1, f) == 1;
    if (!tensor_descs.empty()) {
        ok = ok && std::fwrite(tensor_descs.data(), sizeof(TensorDesc), tensor_descs.size(), f) == tensor_descs.size();
    }
    if (!node_descs.empty()) {
        ok = ok && std::fwrite(node_descs.data(), sizeof(NodeDesc), node_descs.size(), f) == node_descs.size();
    }
    if (!weights_blob.empty()) {
        ok = ok && std::fwrite(weights_blob.data(), 1, weights_blob.size(), f) == weights_blob.size();
    }
    std::fclose(f);
    return ok;
}

}  // namespace tge

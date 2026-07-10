"""Round-trip coverage for the byte-exact TGEM binary artifact format
(src/model_format.py). New test file (no direct C++ equivalent) added
alongside the Python port to document and pin the wire format.
"""

from __future__ import annotations

from pathlib import Path

from src import demo_graph, model_format


def test_struct_layout_itemsizes():
    """Executable documentation of the wire format: these sizes are exactly
    the C++ FileHeader/TensorDesc/NodeDesc struct sizes."""
    assert model_format.FILE_HEADER_DTYPE.itemsize == 64
    assert model_format.TENSOR_DESC_DTYPE.itemsize == 48
    assert model_format.NODE_DESC_DTYPE.itemsize == 24


def test_write_then_parse_roundtrip(tmp_path):
    artifact_path = tmp_path / "roundtrip.tge"
    result = demo_graph.build(str(artifact_path), batch=8, seed=7)

    buffer = Path(artifact_path).read_bytes()
    view = model_format.parse(buffer)

    assert bytes(view.header["magic"]) == model_format.MAGIC
    assert int(view.header["version"]) == model_format.FORMAT_VERSION
    assert int(view.header["num_tensors"]) == result.num_tensors == 21
    assert int(view.header["num_nodes"]) == result.num_nodes == 14
    assert int(view.header["arena_size_bytes"]) == result.arena_size_bytes == 39904
    assert int(view.header["weights_blob_size_bytes"]) == result.weights_blob_size_bytes
    assert int(view.header["alignment"]) == 16

    assert len(view.tensors) == result.num_tensors
    assert len(view.nodes) == result.num_nodes
    assert len(view.weights_blob) == result.weights_blob_size_bytes

    # Every tensor id round-trips to its own array index (GraphBuilder
    # assigns sequential ids, and write_artifact places tensor_descs[t.id]).
    for i, desc in enumerate(view.tensors):
        assert int(desc["id"]) == i

    # Node output ids are all valid tensor ids, and unused input slots are
    # the KINVALID_ID sentinel.
    from src.types import KINVALID_ID

    for node in view.nodes:
        assert 0 <= int(node["output_id"]) < result.num_tensors
        for input_id in node["input_ids"]:
            assert input_id == KINVALID_ID or 0 <= int(input_id) < result.num_tensors


def test_parse_rejects_bad_magic():
    import numpy as np
    import pytest

    header = np.zeros(1, dtype=model_format.FILE_HEADER_DTYPE)[0]
    header["magic"] = b"XXXX"
    header["version"] = model_format.FORMAT_VERSION
    with pytest.raises(ValueError, match="bad magic"):
        model_format.parse(header.tobytes())


def test_parse_rejects_bad_version():
    import numpy as np
    import pytest

    header = np.zeros(1, dtype=model_format.FILE_HEADER_DTYPE)[0]
    header["magic"] = model_format.MAGIC
    header["version"] = model_format.FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="unsupported artifact version"):
        model_format.parse(header.tobytes())

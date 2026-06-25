"""Shape/dtype checks for the IMDb HeteroData builder.

Deliberately no accuracy assertions: the sample data has only 7 labeled
movies, far too few to support a meaningful accuracy threshold. These tests
just confirm the export -> HeteroData bridge produces a well-formed graph.

Requires results/gnn_export/ to already exist (run ./run_all.sh and
src/python/export_features_for_gnn.py first, or run scripts/run_gnn_pipeline.sh).
"""

import pytest
import torch

from gnn.data.imdb_hetero_data import build_imdb_hetero_data

EXPORT_DIR = "results/gnn_export"


@pytest.fixture(scope="module")
def data():
    try:
        return build_imdb_hetero_data(EXPORT_DIR)
    except FileNotFoundError:
        pytest.skip(f"{EXPORT_DIR} not found -- run ./run_all.sh && "
                    "python3 src/python/export_features_for_gnn.py first")


def test_node_types_present(data):
    assert set(data.node_types) == {"actor", "movie"}


def test_edge_types_present(data):
    expected = {
        ("actor", "co_appeared_with", "actor"),
        ("movie", "similar_to", "movie"),
        ("actor", "acted_in", "movie"),
        ("movie", "performed_by", "actor"),
    }
    assert expected.issubset(set(data.edge_types))


def test_feature_shapes_match_node_counts(data):
    assert data["actor"].x.size(0) == data["actor"].num_nodes
    assert data["movie"].x.size(0) == data["movie"].num_nodes
    assert data["movie"].y.size(0) == data["movie"].num_nodes
    assert data["movie"].label_mask.size(0) == data["movie"].num_nodes


def test_rating_is_not_a_feature(data):
    # The label must never be concatenated into the movie feature matrix.
    num_features = data["movie"].x.size(1)
    rating_values = set(round(v, 1) for v in data["movie"].y[data["movie"].label_mask].tolist())
    for col in range(num_features):
        col_values = set(round(v, 1) for v in data["movie"].x[:, col].tolist())
        assert not (rating_values and rating_values.issubset(col_values) and len(rating_values) > 1), \
            "a movie feature column looks suspiciously like the rating label"


def test_label_mask_matches_finite_labels(data):
    mask = data["movie"].label_mask
    y = data["movie"].y
    assert torch.isfinite(y[mask]).all()
    assert torch.isnan(y[~mask]).all()


def test_demo_movies_resolved(data):
    assert len(data.demo_movies) == 3
    for movie in data.demo_movies:
        assert 0 <= movie["index"] < data["movie"].num_nodes


def test_edge_index_within_bounds(data):
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        edge_index = data[edge_type].edge_index
        if edge_index.numel() == 0:
            continue
        assert edge_index[0].max() < data[src_type].num_nodes
        assert edge_index[1].max() < data[dst_type].num_nodes

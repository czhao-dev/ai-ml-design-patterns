"""Shape/dtype checks for the MovieLens HeteroData builder.

Uses a small synthetic fixture (tests/fixtures/movielens_tiny/) rather than
the real ml-latest-small download, so this runs without network access and
without any MovieLens licensing concern (it's not real MovieLens data).
"""

from pathlib import Path

import torch

from gnn.data.movielens_hetero_data import build_movielens_hetero_data

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "movielens_tiny"


def _build(**kwargs):
    return build_movielens_hetero_data(FIXTURE_DIR, min_ratings_per_user=4, **kwargs)


def test_node_types_present():
    data = _build()
    assert set(data.node_types) == {"user", "movie"}


def test_movie_feature_shape_matches_node_count():
    data = _build()
    assert data["movie"].x.size(0) == data["movie"].num_nodes


def test_train_val_test_edges_are_disjoint():
    data = _build(split_strategy="leave_one_out")
    rel = data["user", "rates", "movie"]

    def edge_set(edge_label_index):
        return set(map(tuple, edge_label_index.t().tolist()))

    train_edges = edge_set(rel.train_edge_label_index)
    val_edges = edge_set(rel.val_edge_label_index)
    test_edges = edge_set(rel.test_edge_label_index)
    assert train_edges.isdisjoint(val_edges)
    assert train_edges.isdisjoint(test_edges)
    assert val_edges.isdisjoint(test_edges)


def test_message_passing_edges_only_contain_train_split():
    data = _build(split_strategy="leave_one_out")
    rel = data["user", "rates", "movie"]
    assert torch.equal(rel.edge_index, rel.train_edge_label_index)


def test_reverse_edge_is_flipped_train_edges():
    data = _build()
    forward = data["user", "rates", "movie"].edge_index
    reverse = data["movie", "rated_by", "user"].edge_index
    assert torch.equal(reverse, forward.flip(0))


def test_random_holdout_split_strategy_runs():
    data = _build(split_strategy="random_holdout")
    rel = data["user", "rates", "movie"]
    assert rel.train_edge_label_index.size(1) > 0


def test_eval_meta_present():
    data = _build()
    assert "train_movies_by_user" in data.eval_meta
    assert data.eval_meta["num_users"] == data["user"].num_nodes
    assert data.eval_meta["num_movies"] == data["movie"].num_nodes

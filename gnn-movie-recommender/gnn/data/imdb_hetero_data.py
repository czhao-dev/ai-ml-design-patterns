"""Build a PyTorch Geometric HeteroData object from the igraph-stage exports.

Reads only the flat files written by src/python/export_features_for_gnn.py
(RESULTS_DIR/gnn_export/) -- never the igraph pickles directly, so this code
never has to unpickle something written by a different Python/numpy version
than the one running PyTorch.

Node types: "actor", "movie".
Edge types:
    ("actor", "co_appeared_with", "actor")  -- weighted, both directions
        already present in the export (see build_actor_movie_dicts.py).
    ("movie", "similar_to", "movie")        -- weighted, Jaccard similarity;
        the export lists each pair once, so both directions are added here.
    ("actor", "acted_in", "movie") / ("movie", "performed_by", "actor")
        -- unweighted bipartite cast structure.

Movie rating is exposed as `data["movie"].y` (NaN where unlabeled) plus a
boolean `data["movie"].label_mask`, and is never folded into node features.
"""

import csv
import json
from pathlib import Path

import torch
from torch_geometric.data import HeteroData


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_edge_csv(path, n_cols):
    """Lean reader for the (potentially multi-million-row) edge CSVs -- avoids
    csv.DictReader's per-row dict allocation, which is the main memory/time
    cost at full IMDb scale (tens of millions of edges)."""
    columns = [[] for _ in range(n_cols)]
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            for col, value in zip(columns, row):
                col.append(value)
    return columns


def _zscore(values):
    t = torch.tensor(values, dtype=torch.float)
    std = t.std()
    if std < 1e-8:
        return torch.zeros_like(t)
    return (t - t.mean()) / std


def _one_hot_vocab(values, vocab):
    index = {v: i for i, v in enumerate(vocab)}
    out = torch.zeros(len(values), len(vocab))
    for row, v in enumerate(values):
        out[row, index[v]] = 1.0
    return out


def build_imdb_hetero_data(export_dir):
    """export_dir is RESULTS_DIR/gnn_export from export_features_for_gnn.py."""
    export_dir = Path(export_dir)

    with open(export_dir / "meta.json") as f:
        meta = json.load(f)
    num_actors, num_movies = meta["num_actors"], meta["num_movies"]

    # ---- Actor node features ----
    actor_rows = sorted(_read_csv(export_dir / "actor_features.csv"), key=lambda r: int(r["actor_index"]))
    pagerank = _zscore([float(r["pagerank_score"]) for r in actor_rows])
    degree = _zscore([torch.log1p(torch.tensor(float(r["degree"]))).item() for r in actor_rows])
    is_collaborator = torch.tensor([float(r["is_top100_director_collaborator"]) for r in actor_rows])
    actor_x = torch.stack([pagerank, degree, is_collaborator], dim=1)

    # ---- Movie node features ----
    movie_rows = sorted(_read_csv(export_dir / "movie_features.csv"), key=lambda r: int(r["movie_index"]))
    genre_vocab = meta["genre_vocab"]
    genre_onehot = _one_hot_vocab([r["genre"] for r in movie_rows], genre_vocab)
    community_vocab = sorted({r["community_id"] for r in movie_rows}, key=int)
    community_onehot = _one_hot_vocab([r["community_id"] for r in movie_rows], community_vocab)
    director_flag = torch.tensor([float(r["director_top100_flag"]) for r in movie_rows])
    num_credited = _zscore([torch.log1p(torch.tensor(float(r["num_credited_actors"]))).item() for r in movie_rows])
    movie_x = torch.cat([
        genre_onehot,
        community_onehot,
        director_flag.unsqueeze(1),
        num_credited.unsqueeze(1),
    ], dim=1)

    # ---- Movie labels (rating regression target) ----
    y = torch.full((num_movies,), float("nan"))
    label_mask = torch.zeros(num_movies, dtype=torch.bool)
    for row in _read_csv(export_dir / "movie_labels.csv"):
        idx = int(row["movie_index"])
        y[idx] = float(row["rating"])
        label_mask[idx] = True

    data = HeteroData()
    data["actor"].x = actor_x
    data["movie"].x = movie_x
    data["movie"].y = y
    data["movie"].label_mask = label_mask
    data["movie"].num_nodes = num_movies
    data["actor"].num_nodes = num_actors

    # ---- actor <-> actor (co-appearance; both directions already in the export) ----
    aa_src, aa_dst, aa_weight = _read_edge_csv(export_dir / "edges_actor_actor.csv", 3)
    if aa_src:
        src = torch.tensor([int(v) for v in aa_src])
        dst = torch.tensor([int(v) for v in aa_dst])
        weight = torch.tensor([float(v) for v in aa_weight])
    else:
        src = dst = torch.empty(0, dtype=torch.long)
        weight = torch.empty(0)
    data["actor", "co_appeared_with", "actor"].edge_index = torch.stack([src, dst])
    data["actor", "co_appeared_with", "actor"].edge_weight = weight

    # ---- movie <-> movie (Jaccard similarity; export lists each pair once) ----
    mm_src, mm_dst, mm_weight = _read_edge_csv(export_dir / "edges_movie_movie.csv", 3)
    if mm_src:
        src = torch.tensor([int(v) for v in mm_src])
        dst = torch.tensor([int(v) for v in mm_dst])
        weight = torch.tensor([float(v) for v in mm_weight])
        src2 = torch.cat([src, dst])
        dst2 = torch.cat([dst, src])
        weight2 = torch.cat([weight, weight])
    else:
        src2 = dst2 = torch.empty(0, dtype=torch.long)
        weight2 = torch.empty(0)
    data["movie", "similar_to", "movie"].edge_index = torch.stack([src2, dst2])
    data["movie", "similar_to", "movie"].edge_weight = weight2

    # ---- actor <-> movie bipartite cast structure ----
    am_actor, am_movie = _read_edge_csv(export_dir / "edges_actor_movie.csv", 2)
    actor_idx = torch.tensor([int(v) for v in am_actor])
    movie_idx = torch.tensor([int(v) for v in am_movie])
    data["actor", "acted_in", "movie"].edge_index = torch.stack([actor_idx, movie_idx])
    data["movie", "performed_by", "actor"].edge_index = torch.stack([movie_idx, actor_idx])

    data.demo_movies = meta["demo_movies"]
    return data

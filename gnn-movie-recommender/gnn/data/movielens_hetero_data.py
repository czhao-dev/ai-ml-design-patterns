"""Build a PyTorch Geometric HeteroData object from MovieLens ratings.csv/movies.csv.

Unlike the IMDb track, this graph has real users, so it can support genuine
personalized top-N recommendation (see gnn/metrics/ranking.py) rather than
just rating regression.

Node types: "user" (no side features -- encoded via a learned nn.Embedding,
see HeteroGNNEncoder), "movie" (multi-hot genre + z-scored release year +
mean/num rating computed from TRAIN edges only, to avoid leaking test
ratings into a node feature).

Edge type ("user", "rates", "movie") carries only the TRAIN split in
edge_index (message passing); val/test ratings are held out of message
passing entirely and exposed only as supervision pairs
(train_edge_label_index/train_edge_label, val_..., test_...) so the model
can never see the exact edge it is asked to predict.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from gnn.data.splits import per_user_leave_one_out_edge_split, per_user_random_holdout_edge_split

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def _zscore(series):
    std = series.std()
    if not std or std < 1e-8:
        return (series * 0).astype(float)
    return (series - series.mean()) / std


def build_movielens_hetero_data(data_dir, split_strategy="leave_one_out", min_ratings_per_user=5, seed=0):
    data_dir = Path(data_dir)
    ratings = pd.read_csv(data_dir / "ratings.csv")
    movies = pd.read_csv(data_dir / "movies.csv")

    eligible_users = ratings.groupby("userId").size()
    eligible_users = eligible_users[eligible_users >= min_ratings_per_user].index
    ratings = ratings[ratings["userId"].isin(eligible_users)].reset_index(drop=True)

    user_ids = sorted(ratings["userId"].unique())
    movie_ids = sorted(movies["movieId"].unique())
    user_id2idx = {uid: i for i, uid in enumerate(user_ids)}
    movie_id2idx = {mid: i for i, mid in enumerate(movie_ids)}
    num_users, num_movies = len(user_ids), len(movie_ids)

    ratings = ratings[ratings["movieId"].isin(movie_id2idx)].reset_index(drop=True)
    user_idx = ratings["userId"].map(user_id2idx).to_numpy()
    movie_idx = ratings["movieId"].map(movie_id2idx).to_numpy()
    rating_vals = ratings["rating"].to_numpy(dtype="float32")
    timestamps = ratings["timestamp"].to_numpy() if "timestamp" in ratings.columns else range(len(ratings))

    if split_strategy == "leave_one_out":
        train_idx, val_idx, test_idx = per_user_leave_one_out_edge_split(user_idx.tolist(), list(timestamps))
    elif split_strategy == "random_holdout":
        train_idx, val_idx, test_idx = per_user_random_holdout_edge_split(user_idx.tolist(), seed=seed)
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}")

    def to_edge_label(idx_list):
        idx_list = list(idx_list)
        edge_label_index = torch.from_numpy(np.stack([user_idx[idx_list], movie_idx[idx_list]])).long()
        edge_label = torch.from_numpy(rating_vals[idx_list]).float()
        return edge_label_index, edge_label

    train_label_index, train_label = to_edge_label(train_idx)
    val_label_index, val_label = to_edge_label(val_idx)
    test_label_index, test_label = to_edge_label(test_idx)

    # ---- Movie features (mean/num rating computed from TRAIN edges only) ----
    genres_series = movies.set_index("movieId").loc[movie_ids, "genres"].fillna("(no genres listed)")
    genre_vocab = sorted({g for genres in genres_series for g in genres.split("|")})
    genre_index = {g: i for i, g in enumerate(genre_vocab)}
    genre_onehot = torch.zeros(num_movies, len(genre_vocab))
    for row, genres in enumerate(genres_series):
        for g in genres.split("|"):
            genre_onehot[row, genre_index[g]] = 1.0

    titles = movies.set_index("movieId").loc[movie_ids, "title"]
    years = titles.map(lambda t: float(YEAR_RE.search(t).group(1)) if YEAR_RE.search(t) else float("nan"))
    years = years.fillna(years.mean())
    release_year = torch.tensor(_zscore(years).to_numpy(dtype="float32"))

    train_movie_idx = movie_idx[train_idx]
    train_ratings = rating_vals[train_idx]
    mean_rating = torch.zeros(num_movies)
    num_ratings = torch.zeros(num_movies)
    sums, counts = {}, {}
    for m, r in zip(train_movie_idx, train_ratings):
        sums[m] = sums.get(m, 0.0) + r
        counts[m] = counts.get(m, 0) + 1
    for m in range(num_movies):
        if m in counts:
            mean_rating[m] = float(sums[m] / counts[m])
            num_ratings[m] = counts[m]
    global_mean = float(train_ratings.mean()) if len(train_ratings) else 0.0
    mean_rating[num_ratings == 0] = global_mean
    num_ratings_log = torch.log1p(num_ratings)
    num_ratings_z = _zscore(pd.Series(num_ratings_log.numpy()))
    num_ratings_z = torch.tensor(num_ratings_z.to_numpy(dtype="float32"))

    movie_x = torch.cat([
        genre_onehot,
        release_year.unsqueeze(1),
        mean_rating.unsqueeze(1),
        num_ratings_z.unsqueeze(1),
    ], dim=1)

    data = HeteroData()
    data["user"].num_nodes = num_users
    data["movie"].x = movie_x
    data["movie"].num_nodes = num_movies

    data["user", "rates", "movie"].edge_index = train_label_index
    data["movie", "rated_by", "user"].edge_index = train_label_index.flip(0)
    data["user", "rates", "movie"].train_edge_label_index = train_label_index
    data["user", "rates", "movie"].train_edge_label = train_label
    data["user", "rates", "movie"].val_edge_label_index = val_label_index
    data["user", "rates", "movie"].val_edge_label = val_label
    data["user", "rates", "movie"].test_edge_label_index = test_label_index
    data["user", "rates", "movie"].test_edge_label = test_label

    # Per-user train movie sets, for excluding already-rated movies from
    # top-N candidate ranking at evaluation time.
    train_movies_by_user = {}
    for u, m in zip(user_idx[train_idx], movie_idx[train_idx]):
        train_movies_by_user.setdefault(int(u), set()).add(int(m))
    data.eval_meta = {
        "num_users": num_users,
        "num_movies": num_movies,
        "train_movies_by_user": train_movies_by_user,
        "movie_id2idx": movie_id2idx,
        "user_id2idx": user_id2idx,
    }
    return data

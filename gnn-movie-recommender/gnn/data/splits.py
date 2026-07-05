"""Split helpers for both tracks.

IMDb track: transductive node-label masking -- the graph is always fully
visible, only which movie *labels* are used for the loss/metrics changes.
MovieLens track: per-user edge masking on the (user, rates, movie) relation.
"""

import random


def leave_one_out_label_splits(label_indices, demo_indices=()):
    """Yield (train_idx, test_idx) pairs, one per labeled node, leaving it out.

    Used for the IMDb sample, where there are too few labeled movies (single
    digits) for a fixed holdout split to be statistically meaningful. Demo
    movies are still evaluated like any other point -- LOO already holds out
    each point in turn.
    """
    label_indices = list(label_indices)
    for held_out in label_indices:
        train_idx = [i for i in label_indices if i != held_out]
        yield train_idx, [held_out]


def holdout_label_split(label_indices, train_frac=0.7, val_frac=0.15, seed=0):
    """Fixed random train/val/test split of labeled node indices.

    Used for the full-scale IMDb config, where there are enough labeled
    movies for a held-out split to be meaningful (unlike the tiny sample).
    """
    label_indices = list(label_indices)
    rng = random.Random(seed)
    rng.shuffle(label_indices)
    n = len(label_indices)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = label_indices[:n_train]
    val_idx = label_indices[n_train:n_train + n_val]
    test_idx = label_indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


def per_user_leave_one_out_edge_split(user_ids, timestamps):
    """Split rating-edge indices per user by recency.

    Returns (train_idx, val_idx, test_idx): for each user with >=3 ratings,
    the most recent rating is test, the second-most-recent is val, and the
    rest are train. Users with fewer than 3 ratings go entirely to train
    (too little signal to hold anything out).
    """
    by_user = {}
    for edge_idx, (user_id, ts) in enumerate(zip(user_ids, timestamps)):
        by_user.setdefault(user_id, []).append((ts, edge_idx))

    train_idx, val_idx, test_idx = [], [], []
    for edges in by_user.values():
        edges.sort(key=lambda x: x[0])
        if len(edges) < 3:
            train_idx.extend(idx for _, idx in edges)
            continue
        train_idx.extend(idx for _, idx in edges[:-2])
        val_idx.append(edges[-2][1])
        test_idx.append(edges[-1][1])
    return train_idx, val_idx, test_idx


def per_user_random_holdout_edge_split(user_ids, train_frac=0.8, val_frac=0.1, seed=0):
    """Simpler random per-user edge split, used for the smoke test / sanity check."""
    by_user = {}
    for edge_idx, user_id in enumerate(user_ids):
        by_user.setdefault(user_id, []).append(edge_idx)

    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []
    for edges in by_user.values():
        edges = list(edges)
        rng.shuffle(edges)
        n = len(edges)
        n_train = max(1, int(n * train_frac)) if n >= 3 else n
        n_val = int(n * val_frac) if n >= 3 else 0
        train_idx.extend(edges[:n_train])
        val_idx.extend(edges[n_train:n_train + n_val])
        test_idx.extend(edges[n_train + n_val:])
    return train_idx, val_idx, test_idx

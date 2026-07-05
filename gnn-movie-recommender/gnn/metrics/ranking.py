"""Precision@K / Recall@K / NDCG@K for the MovieLens top-N recommendation task.

These rank against the *full* movie catalog per user (not sampled negatives)
-- more rigorous, and cheap enough at ml-latest-small's scale (~9.7k movies).
Not numerically comparable to papers that report against sampled negatives.
"""

import math


def _dcg(relevances):
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def precision_at_k(ranked_relevant_flags, k):
    top_k = ranked_relevant_flags[:k]
    return sum(top_k) / k


def recall_at_k(ranked_relevant_flags, k, num_relevant):
    if num_relevant == 0:
        return 0.0
    top_k = ranked_relevant_flags[:k]
    return sum(top_k) / num_relevant


def ndcg_at_k(ranked_relevant_flags, k):
    top_k = ranked_relevant_flags[:k]
    dcg = _dcg(top_k)
    ideal = sorted(top_k, reverse=True)
    idcg = _dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_ranking_for_user(scored_movies, relevant_movie_ids, k_values):
    """scored_movies: list of (movie_id, score), already covering the full
    candidate catalog for this user. relevant_movie_ids: set of movie ids
    the user actually likes (rating >= relevance_threshold) among held-out
    movies. Returns {k: {"precision":..., "recall":..., "ndcg":...}}."""
    ranked = sorted(scored_movies, key=lambda x: x[1], reverse=True)
    relevant_flags = [1 if movie_id in relevant_movie_ids else 0 for movie_id, _ in ranked]
    num_relevant = len(relevant_movie_ids)

    results = {}
    for k in k_values:
        results[k] = {
            "precision": precision_at_k(relevant_flags, k),
            "recall": recall_at_k(relevant_flags, k, num_relevant),
            "ndcg": ndcg_at_k(relevant_flags, k),
        }
    return results

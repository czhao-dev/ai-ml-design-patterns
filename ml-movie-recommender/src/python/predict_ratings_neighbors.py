"""Predict each demo movie's rating by averaging the ratings of its nearest
neighbors (by movie-similarity edge weight) that fall in the same Fast Greedy
community.

Ported from the original src/r/predict_ratings_neighbors.R (igraph in R) to
python-igraph; behavior is unchanged.

Usage: python3 predict_ratings_neighbors.py
  Reads RESULTS_DIR (default "results") from the environment, same as the
  rest of the Python pipeline.

Inputs (in RESULTS_DIR):
    pickles/community_detection_state.pcy (written by detect_communities_and_neighbors.py)
    tables/movie_id_rating.txt

Outputs (in RESULTS_DIR):
    predictions/neighbor_averaging_predictions.txt
"""

import os
import pickle
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
STATE_PATH = RESULTS_DIR / "pickles" / "community_detection_state.pcy"
RATING_PATH = RESULTS_DIR / "tables" / "movie_id_rating.txt"
OUTPUT_PATH = RESULTS_DIR / "predictions" / "neighbor_averaging_predictions.txt"

with open(STATE_PATH, "rb") as f:
    state = pickle.load(f)

membership = state["membership"]
demo_movies = state["demo_movies"]

# Maximum number of candidate neighbors to consider (by edge weight) and the
# number of same-community neighbors to average ratings over. On the full
# ~250K-movie dataset these were fixed at 500 and 50; on a small sample both
# are auto-clamped below to whatever is actually available.
CANDIDATE_POOL_SIZE = 500
NUM_NEIGHBORS = 50

# ---- Load movie ratings ----

rating_by_name = {}
with open(RATING_PATH) as f:
    for line in f:
        name, rating = line.rstrip("\n").split("\t", 1)
        rating_by_name[name] = float(rating)


# ---- Predict a movie's rating by averaging its same-community neighbors ----

def predict_rating_from_neighbors(movie_id, movie_name, sorted_edges):
    community = membership[movie_id]
    pool_size = min(CANDIDATE_POOL_SIZE, len(sorted_edges))

    if pool_size == 0:
        print(f"{movie_name} (id={movie_id}): no neighbors in the movie-similarity graph; cannot predict.\n")
        return None

    # For each candidate edge, the "neighbor" is whichever endpoint isn't movie_id.
    candidates = [
        e[1] if e[0] == movie_id else e[0]
        for e in sorted_edges[:pool_size]
    ]

    valid_candidates = [
        c for c in candidates
        if membership.get(c) == community and c in rating_by_name
    ]

    num_neighbors = min(NUM_NEIGHBORS, len(valid_candidates))
    if num_neighbors == 0:
        print(f"{movie_name} (id={movie_id}): no same-community neighbors with ratings; cannot predict.\n")
        return None

    neighbor_ids = valid_candidates[:num_neighbors]
    ratings = [rating_by_name[n] for n in neighbor_ids]
    predicted = sum(ratings) / len(ratings)
    print(f"{movie_name} (id={movie_id}): predicted rating = {predicted:.4f} "
          f"(averaged over {num_neighbors} same-community neighbor(s): {', '.join(neighbor_ids)})\n")
    return {"rating": predicted, "neighbors": neighbor_ids}


print("Predicting ratings from same-community neighbor averages:\n")
results = [
    predict_rating_from_neighbors(movie["id"], movie["name"], movie["edges"])
    for movie in demo_movies
]

# ---- Write predictions to file ----

with open(OUTPUT_PATH, "w") as out:
    for movie, result in zip(demo_movies, results):
        if result is None:
            out.write(f"{movie['name']} (id={movie['id']})\tNA\n")
        else:
            neighbors = ",".join(result["neighbors"])
            out.write(f"{movie['name']} (id={movie['id']})\t{result['rating']:.4f}\tneighbors={neighbors}\n")

print(f"Done. Wrote {OUTPUT_PATH}.")

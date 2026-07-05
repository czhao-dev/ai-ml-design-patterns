"""Build the movie-movie similarity graph, run Fast Greedy Newman community
detection, label each community with its dominant genre (if genre data is
available yet), and find each demo movie's top edge-weight neighbors.

Ported from the original src/r/detect_communities_and_neighbors.R (igraph in
R) to python-igraph; behavior is unchanged.

Usage: python3 detect_communities_and_neighbors.py
  Reads RESULTS_DIR (default "results") from the environment, same as the
  rest of the Python pipeline.

Inputs (in RESULTS_DIR):
    edgelists/movie_edgelist.txt
    tables/hashtable_movie_name2index.txt
    tables/movie_id_genre.txt (optional; skipped if missing)

Outputs (in RESULTS_DIR):
    predictions/neighbor_analysis.txt
    pickles/community_detection_state.pcy (handoff to predict_ratings_neighbors.py)
"""

import os
import pickle
from collections import Counter
from pathlib import Path

import igraph as ig


def ensure_dirs(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def detect_communities(graph, method="fast_greedy", weights=None):
    """Run igraph community detection, returning a VertexClustering."""
    if method == "fast_greedy":
        return graph.as_undirected().community_fastgreedy(weights=weights).as_clustering()
    if method == "edge_betweenness":
        return graph.as_undirected().community_edge_betweenness(weights=weights).as_clustering()
    if method == "infomap":
        return graph.community_infomap(edge_weights=weights)
    if method == "walktrap":
        return graph.community_walktrap(weights=weights).as_clustering()
    raise ValueError(f"Unknown community detection method: {method}")


RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
EDGELIST_PATH = RESULTS_DIR / "edgelists" / "movie_edgelist.txt"
MOVIE_INDEX_PATH = RESULTS_DIR / "tables" / "hashtable_movie_name2index.txt"
MOVIE_ID_GENRE_PATH = RESULTS_DIR / "tables" / "movie_id_genre.txt"
STATE_PATH = RESULTS_DIR / "pickles" / "community_detection_state.pcy"
NEIGHBOR_OUTPUT_PATH = RESULTS_DIR / "predictions" / "neighbor_analysis.txt"

ensure_dirs(NEIGHBOR_OUTPUT_PATH.parent, STATE_PATH.parent)

DEMO_MOVIES = [
    "Batman v Superman: Dawn of Justice (2016)",
    "Mission: Impossible - Rogue Nation (2015)",
    "Minions (2015)",
]

# ---- Build the movie-movie graph and detect communities ----

g = ig.Graph.Read_Ncol(str(EDGELIST_PATH), names=True, weights=True, directed=False)
fgc = detect_communities(g, method="fast_greedy", weights=g.es["weight"])
membership = {g.vs[i]["name"]: comm for i, comm in enumerate(fgc.membership)}

print(f"Graph has {g.vcount()} vertices and {g.ecount()} edges.")
print(f"Fast Greedy community detection found {len(fgc)} communities.\n")

# ---- Look up the canonical movie IDs for the demo movies ----

movie_name2id = {}
with open(MOVIE_INDEX_PATH) as f:
    for line in f:
        idx, name = line.rstrip("\n").split("\t", 1)
        movie_name2id[name] = idx

demo_movie_ids = {name: movie_name2id[name] for name in DEMO_MOVIES}

# ---- Tag each community with its dominant genre (if genre data is available) ----

if MOVIE_ID_GENRE_PATH.exists():
    genre_by_id = {}
    with open(MOVIE_ID_GENRE_PATH) as f:
        for line in f:
            movie_id, genre = line.rstrip("\n").split("\t", 1)
            genre_by_id[movie_id] = genre

    for n, members in enumerate(fgc):
        member_names = [g.vs[v]["name"] for v in members]
        member_genres = [genre_by_id[m] for m in member_names if m in genre_by_id]
        if member_genres:
            counts = Counter(member_genres)
            # Match R's table()+which.max(): ties broken alphabetically by genre name.
            top_genre = max(sorted(counts), key=lambda name: counts[name])
            top_share = counts[top_genre] / len(members)
            print(f"Community {n + 1} ({len(members)} movies): dominant genre = "
                  f"{top_genre} ({100 * top_share:.0f}% of tagged movies)")
        else:
            print(f"Community {n + 1} ({len(members)} movies): no genre data available")
else:
    print(f"Skipping genre tagging: {MOVIE_ID_GENRE_PATH} not found yet.")

# ---- Find each demo movie's top edge-weight neighbors ----

# All edges as (endpoint1_name, endpoint2_name, weight), in file order -- this
# mirrors R's as_edgelist(g)/E(g)$weight, which preserve read order for graphs
# built from an ncol file.
all_edges = [(g.vs[e.source]["name"], g.vs[e.target]["name"], e["weight"]) for e in g.es]


def get_sorted_neighbors(movie_id):
    """Every edge touching movie_id, sorted by weight descending.

    Mirrors R's `union(which(col1==id), which(col2==id))` order: edges where
    movie_id is the first endpoint come before edges where it's the second,
    each group in original file order, before the (stable) sort by weight.
    """
    matches_first = [e for e in all_edges if e[0] == movie_id]
    matches_second = [e for e in all_edges if e[1] == movie_id]
    edges = matches_first + matches_second
    edges.sort(key=lambda e: e[2], reverse=True)
    return edges


def top_n_neighbors(edges, n=5):
    return edges[:n]


sorted_neighbors_by_id = {movie_id: get_sorted_neighbors(movie_id) for movie_id in demo_movie_ids.values()}

print("\nTop neighbors by edge weight (movie_id_1, movie_id_2, weight):")
for name in DEMO_MOVIES:
    movie_id = demo_movie_ids[name]
    print(f"\n{name} (id={movie_id}):")
    for e1, e2, w in top_n_neighbors(sorted_neighbors_by_id[movie_id]):
        print(f"{e1}\t{e2}\t{w:.6f}")

# ---- Write neighbor analysis to file ----

with open(NEIGHBOR_OUTPUT_PATH, "w") as out:
    for name in DEMO_MOVIES:
        movie_id = demo_movie_ids[name]
        top_edges = top_n_neighbors(sorted_neighbors_by_id[movie_id])
        out.write(f"{name} (id={movie_id})\n")
        if not top_edges:
            out.write("  (no neighbors)\n")
        else:
            for e1, e2, w in top_edges:
                out.write(f"  {e1}\t{e2}\t{w:.6f}\n")
        out.write("\n")

# ---- Save state for predict_ratings_neighbors.py ----

state = {
    "membership": membership,
    "demo_movies": [
        {"name": name, "id": demo_movie_ids[name], "edges": sorted_neighbors_by_id[demo_movie_ids[name]]}
        for name in DEMO_MOVIES
    ],
}
with open(STATE_PATH, "wb") as f:
    pickle.dump(state, f)

print(f"\nDone. Wrote {NEIGHBOR_OUTPUT_PATH} and {STATE_PATH}.")

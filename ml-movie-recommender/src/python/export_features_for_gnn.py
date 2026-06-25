"""Export the igraph-stage outputs as flat CSV/JSON files for the GNN stage.

This is a pure exporter: it touches only the standard library, never PyTorch.
That keeps a hard boundary between the two stages -- the igraph venv never
needs torch, and the GNN venv (a separate Python install, see README) never
has to unpickle a file written by a different Python version. The GNN-side
HeteroData builders (gnn/data/imdb_hetero_data.py) read only the files this
script writes, never the pickles in RESULTS_DIR/pickles directly.

Run this after run_all.sh has populated RESULTS_DIR (default "results").

Inputs (in RESULTS_DIR / DATA_DIR):
    pickles/act2movie_dict.pcy, movie2act_dict.pcy
    pickles/hashtable_act_name2index.pcy, hashtable_movie_name2index.pcy
    pickles/sorted_pagerank_scores.pcy
    pickles/community_detection_state.pcy (optional; written by pass 2 of
        detect_communities_and_neighbors.py -- if absent, all movies get
        community_id = -1)
    tables/hashtable_act_name2index.txt, hashtable_movie_name2index.txt
    tables/movie_id_genre.txt, movie_id_rating.txt
    edgelists/act_edgelist.txt, movie_edgelist.txt
    DATA_DIR/director_top100.txt, director_movies.txt

Outputs (in RESULTS_DIR/gnn_export):
    actor_features.csv   actor_index,pagerank_score,degree,is_top100_director_collaborator
    movie_features.csv   movie_index,num_credited_actors,community_id,genre,director_top100_flag
    movie_labels.csv      movie_index,rating
    edges_actor_actor.csv      src,dst,weight       (directed, co-appearance fraction)
    edges_movie_movie.csv      src,dst,weight       (undirected, Jaccard similarity)
    edges_actor_movie.csv      actor_index,movie_index   (bipartite, unweighted)
    meta.json             counts, demo-movie ids, genre vocabulary
"""

import csv
import json
import os
import pickle
import re
import time
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "data/sample"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
PICKLES_DIR = RESULTS_DIR / "pickles"
TABLES_DIR = RESULTS_DIR / "tables"
EDGELISTS_DIR = RESULTS_DIR / "edgelists"
EXPORT_DIR = RESULTS_DIR / "gnn_export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Same three worked-example movies used throughout the rest of the pipeline.
DEMO_MOVIES = [
    "Batman v Superman: Dawn of Justice (2016)",
    "Mission: Impossible - Rogue Nation (2015)",
    "Minions (2015)",
]

NONAME_MOVIES = {'(2006)', '(1967)', '(1971)', '(1993)', '(1995)', '(1996)', '(2001)', '(2002)',
                 '(2005)', '(2006)', '(2009)', '(2010)', '(2012)', '(2013)', '(2014)'}

start = time.perf_counter()


def read_index_table(path):
    """Read a "<index>\\t<name>\\n" table into index->name and name->index dicts."""
    index2name, name2index = {}, {}
    with open(path) as f:
        for line in f:
            idx, name = line.rstrip("\n").split("\t", 1)
            index2name[int(idx)] = name
            name2index[name] = int(idx)
    return index2name, name2index


print('(Step 1 of 10) Load canonical actor/movie index tables.')
act_index2name, act_name2index = read_index_table(TABLES_DIR / "hashtable_act_name2index.txt")
movie_index2name, movie_name2index = read_index_table(TABLES_DIR / "hashtable_movie_name2index.txt")
num_actors = len(act_index2name)
num_movies = len(movie_index2name)

print('(Step 2 of 10) Load act2movie_dict / movie2act_dict / PageRank scores from pickles.')
with open(PICKLES_DIR / "act2movie_dict.pcy", "rb") as f:
    act2movie_dict = pickle.load(f)
with open(PICKLES_DIR / "movie2act_dict.pcy", "rb") as f:
    movie2act_dict = pickle.load(f)
with open(PICKLES_DIR / "sorted_pagerank_scores.pcy", "rb") as f:
    pagerank_by_index_str = pickle.load(f)

print('(Step 3 of 10) Load community membership, if community detection has run.')
community_state_path = PICKLES_DIR / "community_detection_state.pcy"
membership_by_index_str = {}
if community_state_path.exists():
    with open(community_state_path, "rb") as f:
        membership_by_index_str = pickle.load(f)["membership"]

print('(Step 4 of 10) Load movie genre and rating tables.')
genre_by_index = {}
if (TABLES_DIR / "movie_id_genre.txt").exists():
    with open(TABLES_DIR / "movie_id_genre.txt") as f:
        for line in f:
            idx, genre = line.rstrip("\n").split("\t", 1)
            genre_by_index[int(idx)] = genre

rating_by_index = {}
with open(TABLES_DIR / "movie_id_rating.txt") as f:
    for line in f:
        idx, rating = line.rstrip("\n").split("\t", 1)
        rating_by_index[int(idx)] = float(rating)

print('(Step 5 of 10) Cross-reference director_top100.txt with director_movies.txt '
      '(same logic as predict_ratings_regression.py) to flag top-100-director movies.')
director100_set = set()
with open(DATA_DIR / "director_top100.txt") as f:
    for line in f:
        director100_set.add(line.rstrip("\n"))

director100_movies_set = set()
with open(DATA_DIR / "director_movies.txt") as f:
    for line in f:
        line_split = [s for s in line.split("\t") if s != ""]
        if line_split[0] in director100_set:
            for movie_field in line_split[1:]:
                movie_name = re.sub(r'\([^0-9)]*\)', '', movie_field)
                movie_name = re.sub(r'{{.*?}}', '', movie_name)
                movie_name = movie_name.strip()
                if movie_name not in NONAME_MOVIES:
                    director100_movies_set.add(movie_name)

print('(Step 6 of 10) Compute actor degree from act_edgelist.txt and write actor_features.csv.')
actor_degree = {i: 0 for i in range(num_actors)}
with open(EDGELISTS_DIR / "act_edgelist.txt") as f:
    for line in f:
        src, dst, _weight = line.rstrip("\n").split("\t")
        actor_degree[int(src)] += 1

with open(EXPORT_DIR / "actor_features.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["actor_index", "pagerank_score", "degree", "is_top100_director_collaborator"])
    for idx in range(num_actors):
        pagerank_score = pagerank_by_index_str.get(str(idx), 0.0)
        actor_name = act_index2name[idx]
        actor_movies = act2movie_dict.get(actor_name, [])
        is_top100_collaborator = int(any(m in director100_movies_set for m in actor_movies))
        writer.writerow([idx, pagerank_score, actor_degree[idx], is_top100_collaborator])

print('(Step 7 of 10) Write movie_features.csv (community id, genre, director-top100 flag).')
with open(EXPORT_DIR / "movie_features.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["movie_index", "num_credited_actors", "community_id", "genre", "director_top100_flag"])
    for idx in range(num_movies):
        movie_name = movie_index2name[idx]
        num_credited_actors = len(movie2act_dict.get(movie_name, []))
        community_id = membership_by_index_str.get(str(idx), -1)
        genre = genre_by_index.get(idx, "unknown")
        director_top100_flag = int(movie_name in director100_movies_set)
        writer.writerow([idx, num_credited_actors, community_id, genre, director_top100_flag])

print('(Step 8 of 10) Write movie_labels.csv (rating targets).')
with open(EXPORT_DIR / "movie_labels.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["movie_index", "rating"])
    for idx in sorted(rating_by_index):
        writer.writerow([idx, rating_by_index[idx]])

print('(Step 9 of 10) Copy actor-actor / movie-movie edgelists and derive actor-movie bipartite edges.')
with open(EDGELISTS_DIR / "act_edgelist.txt") as src, \
        open(EXPORT_DIR / "edges_actor_actor.csv", "w", newline="") as dst:
    writer = csv.writer(dst)
    writer.writerow(["src", "dst", "weight"])
    for line in src:
        writer.writerow(line.rstrip("\n").split("\t"))

with open(EDGELISTS_DIR / "movie_edgelist.txt") as src, \
        open(EXPORT_DIR / "edges_movie_movie.csv", "w", newline="") as dst:
    writer = csv.writer(dst)
    writer.writerow(["src", "dst", "weight"])
    for line in src:
        writer.writerow(line.rstrip("\n").split("\t"))

with open(TABLES_DIR / "movie2act_dict.txt") as src, \
        open(EXPORT_DIR / "edges_actor_movie.csv", "w", newline="") as dst:
    writer = csv.writer(dst)
    writer.writerow(["actor_index", "movie_index"])
    for line in src:
        fields = line.rstrip("\n").split("\t")
        movie_name, actor_names = fields[0], fields[1:]
        movie_index = movie_name2index[movie_name]
        for actor_name in actor_names:
            actor_index = act_name2index.get(actor_name)
            if actor_index is not None:
                writer.writerow([actor_index, movie_index])

print('(Step 10 of 10) Resolve demo movie ids and write meta.json.')
demo_movies = [{"name": name, "index": movie_name2index[name]} for name in DEMO_MOVIES if name in movie_name2index]
genre_vocab = sorted(set(genre_by_index.values()) | {"unknown"})
meta = {
    "num_actors": num_actors,
    "num_movies": num_movies,
    "num_labeled_movies": len(rating_by_index),
    "num_communities": len(set(membership_by_index_str.values())) if membership_by_index_str else 0,
    "genre_vocab": genre_vocab,
    "demo_movies": demo_movies,
}
with open(EXPORT_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

end = time.perf_counter()
print(f"\nWrote {EXPORT_DIR}/ ({num_actors} actors, {num_movies} movies, "
      f"{len(rating_by_index)} labeled movies).")
print(f'Process done. Program run-time: {end - start:.2f} seconds.')

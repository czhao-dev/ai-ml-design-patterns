"""Predict movie ratings by averaging actor "quality" scores over a bipartite actor-movie graph.

Each actor's score is the mean of their top-5 movie ratings (or all ratings if they have
fewer than 5), falling back to a neutral 5.0 if none of their movies have a rating.
A movie's predicted rating is the mean score of its credited actors.

Inputs:
    RESULTS_DIR/pickles/act2movie_dict.pcy
    RESULTS_DIR/pickles/movie_rating_dict.pcy (written by predict_ratings_regression.py)

Outputs:
    RESULTS_DIR/predictions/bipartite_predictions.txt
"""

import os
import pickle
import statistics
import time
from pathlib import Path

import igraph

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
PICKLES_DIR = RESULTS_DIR / "pickles"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

TEST_MOVIES = ['Batman v Superman: Dawn of Justice (2016)', 'Mission: Impossible - Rogue Nation (2015)', 'Minions (2015)']

start = time.perf_counter()

print('(Step 1 of 5) Load act2movie_dict from pickle file.')
with open(PICKLES_DIR / 'act2movie_dict.pcy', 'rb') as f:
    act2movie_dict = pickle.load(f)

print('(Step 2 of 5) Load movie_rating_dict from pickle file.')
with open(PICKLES_DIR / 'movie_rating_dict.pcy', 'rb') as f:
    movie_rating_dict = pickle.load(f)

print('(Step 3 of 5) Create edgelist of actor and movie vertices and score each actor.')
types_arr = []
edges_arr = []
act_score_dict = {}
vertex_id = 0
vertex2act_name = {}
movie_name2vertex = {}

for act_name, movies in act2movie_dict.items():
    act_vertex_id = vertex_id
    vertex2act_name[act_vertex_id] = act_name
    types_arr.append(0)
    vertex_id += 1

    act_movie_rating_arr = []
    for movie in movies:
        if movie in movie_rating_dict:
            act_movie_rating_arr.append(movie_rating_dict[movie])

        movie_vertex_id = vertex_id
        movie_name2vertex[movie] = movie_vertex_id
        types_arr.append(1)
        edges_arr.append((act_vertex_id, movie_vertex_id))
        vertex_id += 1

    if len(act_movie_rating_arr) > 5:
        sorted_ratings = sorted(act_movie_rating_arr, reverse=True)
        act_score = 0.9 * statistics.mean(sorted_ratings[0:5])
    elif len(act_movie_rating_arr) != 0:
        act_score = statistics.mean(act_movie_rating_arr)
    else:
        act_score = 5.0
    act_score_dict[act_name] = act_score

print('(Step 4 of 5) Create bipartite graph.')
g = igraph.Graph.Bipartite(types=types_arr, edges=edges_arr, directed=False)
print(f'Graph g has {g.vcount()} vertices and {g.ecount()} edges.')

print('(Step 5 of 5) Predict ratings of the test movies.\n')
with open(PREDICTIONS_DIR / 'bipartite_predictions.txt', 'w') as f:
    for movie_name in TEST_MOVIES:
        movie_vertex_id = movie_name2vertex[movie_name]
        act_names = [vertex2act_name[v] for v in g.neighbors(movie_vertex_id)]
        act_scores = [act_score_dict[name] for name in act_names if act_score_dict[name] != 0]
        predicted_rating = statistics.mean(act_scores)
        line = f"{movie_name}\t{predicted_rating}"
        print(line)
        f.write(line + '\n')

end = time.perf_counter()
print(f'\nProcess done. Program run-time: {end - start:.2f} seconds.')

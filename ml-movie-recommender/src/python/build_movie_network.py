"""Build the movie-movie similarity network (Jaccard index of shared cast members).

Reuses the canonical act2movie_dict / movie2act_dict / hashtable_movie_name2index pickles
produced by build_actor_movie_dicts.py, so movie indices stay consistent across the pipeline.

Inputs (in RESULTS_DIR/pickles, default "results/pickles"):
    act2movie_dict.pcy, movie2act_dict.pcy, hashtable_movie_name2index.pcy

Outputs (in RESULTS_DIR/edgelists, default "results/edgelists"):
    movie_edgelist.txt
        "<movie1_index>\\t<movie2_index>\\t<jaccard>\\n" -- an edge is only written between
        two movies that each have >=5 actors in movie2act_dict.
"""

import itertools
import os
import pickle
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
PICKLES_DIR = RESULTS_DIR / "pickles"
EDGELISTS_DIR = RESULTS_DIR / "edgelists"
EDGELISTS_DIR.mkdir(parents=True, exist_ok=True)

with open(PICKLES_DIR / 'act2movie_dict.pcy', 'rb') as f:
    act2movie_dict = pickle.load(f)
with open(PICKLES_DIR / 'movie2act_dict.pcy', 'rb') as f:
    movie2act_dict = pickle.load(f)
with open(PICKLES_DIR / 'hashtable_movie_name2index.pcy', 'rb') as f:
    hashtable_movie_name2index = pickle.load(f)

sorted_act_names = sorted(act2movie_dict.keys())

print("Constructing movie edgelist...")
movie_combs_dict = {}
with open(EDGELISTS_DIR / 'movie_edgelist.txt', 'w') as output_file:
    for i, act_name in enumerate(sorted_act_names):
        if i % 1000 == 0:
            print(f"  ...{i}/{len(sorted_act_names)} actors processed")

        movie_list = act2movie_dict[act_name]
        if len(movie_list) > 1:
            for movie1_name, movie2_name in itertools.combinations(movie_list, 2):
                movie1_index = hashtable_movie_name2index[movie1_name]
                movie2_index = hashtable_movie_name2index[movie2_name]
                if (movie_combs_dict.get(movie1_index) is None) or (movie2_index not in movie_combs_dict.get(movie1_index)):
                    movie1_actors = movie2act_dict.get(movie1_name)
                    movie2_actors = movie2act_dict.get(movie2_name)
                    if len(movie1_actors) >= 5 and len(movie2_actors) >= 5:
                        overlap_actors = len(set(movie1_actors) & set(movie2_actors))
                        total_actors = len(set(movie1_actors) | set(movie2_actors))

                        entry = [movie1_index, movie2_index, float(overlap_actors) / total_actors]
                        output_file.write("\t".join(str(x) for x in entry) + '\n')

                    movie_combs_dict.setdefault(movie1_index, []).append(movie2_index)
                    movie_combs_dict.setdefault(movie2_index, []).append(movie1_index)

print("Done.")

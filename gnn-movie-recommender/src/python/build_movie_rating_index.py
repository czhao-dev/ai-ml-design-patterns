"""Map movie ratings onto the canonical movie index space.

Inputs:
    DATA_DIR/movie_rating.txt (default "data/sample")
        "<movie_name>\\t\\t<rating>\\n" (double-tab separated)
    RESULTS_DIR/pickles/movie2act_dict.pcy, hashtable_movie_name2index.pcy

Outputs:
    RESULTS_DIR/tables/movie_id_rating.txt
        "<movie_index>\\t<rating>\\n" -- only for movies with >=5 credited actors.
"""

import os
import pickle
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "data/sample"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
PICKLES_DIR = RESULTS_DIR / "pickles"
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

with open(PICKLES_DIR / 'movie2act_dict.pcy', 'rb') as f:
    movie2act_dict = pickle.load(f)
with open(PICKLES_DIR / 'hashtable_movie_name2index.pcy', 'rb') as f:
    hashtable_movie_name2index = pickle.load(f)

print("...Reading movie ratings...")
movie2rating_dict = {}
with open(DATA_DIR / "movie_rating.txt", "r") as f:
    for line in f:
        line_split = [s for s in line.split("\t") if s != '']
        movie_name = line_split[0]
        rating = line_split[1][:-1]
        movie2rating_dict[movie_name] = rating

print("...Constructing movie_id_rating.txt...")
with open(TABLES_DIR / 'movie_id_rating.txt', 'w') as output_file:
    for movie_name, rating in movie2rating_dict.items():
        actors = movie2act_dict.get(movie_name)
        if actors is not None and len(actors) >= 5:
            movie_index = hashtable_movie_name2index[movie_name]
            output_file.write(f"{movie_index}\t{rating}\n")

print("Done.")

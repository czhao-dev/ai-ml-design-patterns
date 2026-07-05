"""Map movie genres onto the canonical movie index space.

Inputs:
    DATA_DIR/movie_genre.txt (default "data/sample")
        "<movie_name>\\t<genre>\\n" (single-tab separated)
    RESULTS_DIR/pickles/movie2act_dict.pcy, hashtable_movie_name2index.pcy

Outputs:
    RESULTS_DIR/tables/movie_id_genre.txt
        "<movie_index>\\t<genre>\\n" -- only for movies with >=5 credited actors. Read by
        detect_communities_and_neighbors.R to label each community's dominant genre.
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

print("...Reading movie genres...")
movie2genre_dict = {}
with open(DATA_DIR / "movie_genre.txt", "r") as f:
    for line in f:
        line_split = [s for s in line.split("\t") if s != '']
        movie_name = line_split[0]
        genre = line_split[1][:-1]
        movie2genre_dict[movie_name] = genre

print("...Constructing movie_id_genre.txt...")
with open(TABLES_DIR / 'movie_id_genre.txt', 'w') as output_file:
    for movie_name, genre in movie2genre_dict.items():
        actors = movie2act_dict.get(movie_name)
        if actors is not None and len(actors) >= 5:
            movie_index = hashtable_movie_name2index[movie_name]
            output_file.write(f"{movie_index}\t{genre}\n")

print("Done.")

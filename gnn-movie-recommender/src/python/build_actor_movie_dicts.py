"""Build actor/actress <-> movie lookup tables and the actor-actor edge list.

Inputs (in DATA_DIR, default "data/sample"):
    actor_movies.txt, actress_movies.txt
        "Name\\tMovie 1\\tMovie 2\\t...\\n" -- rows with fewer than 5 movies are skipped.

Outputs (in RESULTS_DIR, default "results"):
    pickles/act2movie_dict.pcy, pickles/movie2act_dict.pcy
    pickles/hashtable_act_name2index.pcy, pickles/hashtable_movie_name2index.pcy
    tables/hashtable_act_name2index.txt, tables/hashtable_movie_name2index.txt
    tables/movie2act_dict.txt
    edgelists/act_edgelist.txt
        "<actor1_index>\\t<actor2_index>\\t<weight>\\n" where weight is the fraction of
        actor1's movies that actor1 and actor2 share (a directed, asymmetric edge).
"""

import itertools
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

for d in (PICKLES_DIR, TABLES_DIR, EDGELISTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

NONAME_MOVIES = {'(2006)', '(1967)', '(1971)', '(1993)', '(1995)', '(1996)', '(2001)', '(2002)',
                 '(2005)', '(2006)', '(2009)', '(2010)', '(2012)', '(2013)', '(2014)'}


def read_movies_file(filename, act2movie_dict, movie2act_dict):
    with open(filename, "r") as f:
        for line in f:
            line_split = [s for s in line.split("\t") if len(s) > 2]
            line_split = [re.sub(r'[!@#$&.*\']', '', s) for s in line_split]
            if len(line_split) <= 5:
                continue

            person_name = line_split[0]
            movies = []
            for i in range(1, len(line_split)):
                line_split[i] = re.sub(r'\([^0-9)]*\)', '', line_split[i])
                line_split[i] = re.sub(r'{{.*?}}', '', line_split[i])

                while line_split[i].startswith(' '):
                    line_split[i] = line_split[i][1:]
                while line_split[i].endswith(' ') or line_split[i].endswith('\n'):
                    line_split[i] = line_split[i][:-1]

                movie_name = line_split[i]
                if movie_name not in NONAME_MOVIES:
                    movies.append(movie_name)
                    movie2act_dict.setdefault(movie_name, []).append(person_name)

            act2movie_dict[person_name] = movies


start = time.perf_counter()

print('(Step 1 of 8) Read actor_movies.txt and build act_name<->movie_name dictionaries.')
act2movie_dict = {}
movie2act_dict = {}
read_movies_file(DATA_DIR / "actor_movies.txt", act2movie_dict, movie2act_dict)

print('(Step 2 of 8) Read actress_movies.txt and build act_name<->movie_name dictionaries.')
read_movies_file(DATA_DIR / "actress_movies.txt", act2movie_dict, movie2act_dict)

print('(Step 3 of 8) Sort act_names->movie_names and movie_names->act_names dictionaries.')
sorted_act_names = sorted(act2movie_dict.keys())
sorted_movie_names = sorted(movie2act_dict.keys())

print('(Step 4 of 8) Save both dictionaries as pickle files.')
with open(PICKLES_DIR / 'act2movie_dict.pcy', 'wb') as f:
    pickle.dump(act2movie_dict, f)
with open(PICKLES_DIR / 'movie2act_dict.pcy', 'wb') as f:
    pickle.dump(movie2act_dict, f)

print('(Step 5 of 8) Create act_name->index hash table and write to hashtable_act_name2index.txt file.')
hashtable_act_name2index = {}
hashtable_movie_name2index = {}
for i, act_name in enumerate(sorted_act_names):
    hashtable_act_name2index[act_name] = i

with open(TABLES_DIR / 'hashtable_act_name2index.txt', 'w') as f:
    for i, act_name in enumerate(sorted_act_names):
        f.write(f"{i}\t{act_name}\n")

print('(Step 6 of 8) Create movie_name->index hash table and write to hashtable_movie_name2index.txt and movie2act_dict.txt files.')
for i, movie_name in enumerate(sorted_movie_names):
    hashtable_movie_name2index[movie_name] = i

with open(TABLES_DIR / 'hashtable_movie_name2index.txt', 'w') as dict2_file, \
        open(TABLES_DIR / 'movie2act_dict.txt', 'w') as dict3_file:
    for i, movie_name in enumerate(sorted_movie_names):
        dict2_file.write(f"{i}\t{movie_name}\n")

        entry = [movie_name] + movie2act_dict[movie_name]
        dict3_file.write("\t".join(str(x) for x in entry) + '\n')

print('(Step 7 of 8) Save both hash tables as pickle files.')
with open(PICKLES_DIR / 'hashtable_act_name2index.pcy', 'wb') as f:
    pickle.dump(hashtable_act_name2index, f)
with open(PICKLES_DIR / 'hashtable_movie_name2index.pcy', 'wb') as f:
    pickle.dump(hashtable_movie_name2index, f)

print('(Step 8 of 8) Process actors/actresses edge list and save it as act_edgelist.txt file.')
act_combs_dict = {}
with open(EDGELISTS_DIR / 'act_edgelist.txt', 'w') as output_file:
    for movie_name in sorted_movie_names:
        act_list = movie2act_dict.get(movie_name)
        if len(act_list) > 1:
            for act1_name, act2_name in itertools.combinations(act_list, 2):
                act1_index = hashtable_act_name2index[act1_name]
                act2_index = hashtable_act_name2index[act2_name]
                if (act_combs_dict.get(act1_index) is None) or (act2_index not in act_combs_dict.get(act1_index)):
                    act1_movies = act2movie_dict.get(act1_name)
                    act2_movies = act2movie_dict.get(act2_name)
                    overlap_movies = len(set(act1_movies) & set(act2_movies))

                    entry1 = [act1_index, act2_index, float(overlap_movies) / len(act1_movies)]
                    entry2 = [act2_index, act1_index, float(overlap_movies) / len(act2_movies)]
                    output_file.write("\t".join(str(x) for x in entry1) + '\n')
                    output_file.write("\t".join(str(x) for x in entry2) + '\n')

                    act_combs_dict.setdefault(act1_index, []).append(act2_index)
                    act_combs_dict.setdefault(act2_index, []).append(act1_index)

end = time.perf_counter()
print(f'Process done. Program run-time: {end - start:.2f} seconds.')

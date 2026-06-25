"""Compute PageRank for every actor/actress in the actor-actor network.

Inputs (in RESULTS_DIR, default "results"):
    edgelists/act_edgelist.txt
    pickles/movie2act_dict.pcy, pickles/hashtable_act_name2index.pcy

Outputs (in RESULTS_DIR):
    tables/sorted_pagerank_scores.txt, pickles/sorted_pagerank_scores.pcy
        actor index -> PageRank score, sorted ascending.
    tables/movie2act_pr_dict.txt
        "<movie_name>\\t<pr_score_1>\\t<pr_score_2>\\t...\\n" -- PageRank score of every
        actor credited on that movie.
"""

import os
import pickle
import time
from collections import OrderedDict
from pathlib import Path

import igraph

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
PICKLES_DIR = RESULTS_DIR / "pickles"
TABLES_DIR = RESULTS_DIR / "tables"
EDGELISTS_DIR = RESULTS_DIR / "edgelists"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

start = time.perf_counter()

print('(Step 1 of 7) Construct actors/actresses graph from act_edgelist.txt.')
g = igraph.Graph.Read_Ncol(str(EDGELISTS_DIR / 'act_edgelist.txt'), directed=True)
print(f'Graph has {g.vcount()} vertices and {g.ecount()} edges.')

print('(Step 2 of 7) Calculate PageRank for each actor/actress.')
pr = g.pagerank(vertices=g.vs, directed=True, weights=g.es['weight'])

print('(Step 3 of 7) Store PageRank scores in a hash table.')
pr_table = {g.vs['name'][i]: pr[i] for i in range(len(pr))}

print('(Step 4 of 7) Sort PageRank scores.')
sorted_pr_table = OrderedDict(sorted(pr_table.items(), key=lambda x: x[1]))

print('(Step 5 of 7) Save sorted PageRank scores to sorted_pagerank_scores.txt and .pcy.')
with open(TABLES_DIR / 'sorted_pagerank_scores.txt', 'w') as f:
    for k, v in sorted_pr_table.items():
        f.write(f"{k}\t{v}\n")
with open(PICKLES_DIR / 'sorted_pagerank_scores.pcy', 'wb') as f:
    pickle.dump(sorted_pr_table, f)

print('(Step 6 of 7) Load movie2act_dict and hashtable_act_name2index from pickles.')
with open(PICKLES_DIR / 'movie2act_dict.pcy', 'rb') as f:
    movie2act_dict = pickle.load(f)
with open(PICKLES_DIR / 'hashtable_act_name2index.pcy', 'rb') as f:
    hashtable_act_name2index = pickle.load(f)

print('(Step 7 of 7) Look up each movie\'s actors\' PageRank scores and write movie2act_pr_dict.txt.')
with open(TABLES_DIR / 'movie2act_pr_dict.txt', 'w') as f:
    for movie_name, act_names in movie2act_dict.items():
        entry = [movie_name]
        for act_name in act_names:
            act_index = hashtable_act_name2index.get(act_name)
            entry.append(sorted_pr_table.get(str(act_index)))
        f.write("\t".join(str(x) for x in entry) + '\n')

end = time.perf_counter()
print(f'\nProcess done. Program run-time: {end - start:.2f} seconds.')

"""Predict movie ratings with linear regression on actor PageRank features.

Features per movie: the top-5 actor PageRank scores (descending) plus a boolean for
whether the movie's director is in the top-100 director list.

Inputs:
    RESULTS_DIR/pickles/sorted_pagerank_scores.pcy, hashtable_act_name2index.pcy
    RESULTS_DIR/tables/movie2act_dict.txt
    DATA_DIR/director_top100.txt, director_movies.txt, movie_rating.txt

Outputs:
    RESULTS_DIR/pickles/movie2act_pr_dict.pcy, movie_parameters_pickle.pcy,
        movie_ratings_pickle.pcy, movie_rating_dict.pcy, test_movies_pickle.pcy
    RESULTS_DIR/predictions/regression_predictions.txt
"""

import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
from sklearn import linear_model

DATA_DIR = Path(os.environ.get("DATA_DIR", "data/sample"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
PICKLES_DIR = RESULTS_DIR / "pickles"
TABLES_DIR = RESULTS_DIR / "tables"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# The three movies used throughout this project as worked examples.
TEST_MOVIES = ['Batman v Superman: Dawn of Justice (2016)', 'Mission: Impossible - Rogue Nation (2015)', 'Minions (2015)']
TEST_MOVIES_DIRECTORS = ['Snyder, Zack', 'McQuarrie, Christopher', 'Balda, Kyle']

NONAME_MOVIES = {'(2006)', '(1967)', '(1971)', '(1993)', '(1995)', '(1996)', '(2001)', '(2002)',
                 '(2005)', '(2006)', '(2009)', '(2010)', '(2012)', '(2013)', '(2014)'}

start = time.perf_counter()

print('(Step 1 of 14) Read sorted_pagerank_scores from pickle.')
with open(PICKLES_DIR / 'sorted_pagerank_scores.pcy', 'rb') as f:
    sorted_pr_table = pickle.load(f)

print('(Step 2 of 14) Read hashtable_act_name2index from pickle.')
with open(PICKLES_DIR / 'hashtable_act_name2index.pcy', 'rb') as f:
    hashtable_act_name2index = pickle.load(f)

print('(Step 3 of 14) Read movie2act_dict.txt and build top-5 PageRank features per movie.')
movie2act_pr_dict = {}
with open(TABLES_DIR / 'movie2act_dict.txt', 'r') as f:
    for line in f:
        line_split = [s for s in line.split("\t") if s != '']
        line_split[-1] = line_split[-1].rstrip('\n')
        movie_name = line_split[0]
        act_pr_scores = []
        for act_name in line_split[1:]:
            act_index = hashtable_act_name2index.get(act_name)
            act_pr_scores.append(sorted_pr_table.get(str(act_index)))
        act_pr_scores = sorted(act_pr_scores, reverse=True)
        if len(act_pr_scores) >= 5:
            movie2act_pr_dict[movie_name] = act_pr_scores[0:5]

print('(Step 4 of 14) Save movie2act_pr_dict as pickle.')
with open(PICKLES_DIR / 'movie2act_pr_dict.pcy', 'wb') as f:
    pickle.dump(movie2act_pr_dict, f)

print('(Step 5 of 14) Read in top-100 directors.')
director100_set = set()
with open(DATA_DIR / 'director_top100.txt', 'r') as f:
    for line in f:
        director100_set.add(line.rstrip('\n'))

print('(Step 6 of 14) Read in director_movies.txt.')
director100_movies_set = set()
with open(DATA_DIR / 'director_movies.txt', 'r') as f:
    for line in f:
        line_split = [s for s in line.split('\t') if s != '']
        if line_split[0] in director100_set:
            for i in range(1, len(line_split)):
                movie_name = re.sub(r'\([^0-9)]*\)', '', line_split[i])
                movie_name = re.sub(r'{{.*?}}', '', movie_name)
                movie_name = movie_name.strip()
                if movie_name not in NONAME_MOVIES:
                    director100_movies_set.add(movie_name)

print('(Step 7 of 14) Read movie_rating.txt.')
movie_director100_dict = {}  # 1 if the movie's director is in the top-100 list, 0 otherwise
movie_rating_dict = {}       # movie name -> rating
with open(DATA_DIR / 'movie_rating.txt', 'r') as f:
    for line in f:
        line_split = [s for s in line.split('\t\t') if s != '']
        movie_name = line_split[0]
        movie_director100_dict[movie_name] = 1 if movie_name in director100_movies_set else 0
        movie_rating_dict[movie_name] = float(line_split[1].rstrip('\n'))

print('(Step 8 of 14) Create numpy arrays of movie parameters and ratings.')
movie_parameters = []
movie_ratings = []
for movie_name, pr_scores in movie2act_pr_dict.items():
    if movie_name in movie_director100_dict:
        movie_parameters.append(pr_scores + [movie_director100_dict[movie_name]])
        movie_ratings.append(movie_rating_dict.get(movie_name))

print('(Step 9 of 14) Save movie parameters and ratings to pickle.')
movie_parameters_np = np.asarray(movie_parameters)
movie_ratings_np = np.transpose(np.asarray(movie_ratings))
with open(PICKLES_DIR / 'movie_parameters_pickle.pcy', 'wb') as f:
    pickle.dump(movie_parameters_np, f)
with open(PICKLES_DIR / 'movie_ratings_pickle.pcy', 'wb') as f:
    pickle.dump(movie_ratings_np, f)
with open(PICKLES_DIR / 'movie_rating_dict.pcy', 'wb') as f:
    pickle.dump(movie_rating_dict, f)

print('(Step 10 of 14) Collect test movies data.')
test_movies_data = []
for movie_name, director in zip(TEST_MOVIES, TEST_MOVIES_DIRECTORS):
    flag = 1 if director in director100_set else 0
    test_movies_data.append(movie2act_pr_dict[movie_name] + [flag])
test_movies_data_np = np.asarray(test_movies_data)

print('(Step 11 of 14) Save test movies data to pickle.')
with open(PICKLES_DIR / 'test_movies_pickle.pcy', 'wb') as f:
    pickle.dump(test_movies_data_np, f)

print('(Step 12 of 14) Fit linear regression model.')
reg = linear_model.LinearRegression()
reg.fit(movie_parameters_np, movie_ratings_np)

print('(Step 13 of 14) Predict ratings.\n')
predict_scores = reg.predict(test_movies_data_np)
with open(PREDICTIONS_DIR / 'regression_predictions.txt', 'w') as f:
    for movie_name, score in zip(TEST_MOVIES, predict_scores):
        line = f"{movie_name}\t{score}"
        print(line)
        f.write(line + '\n')

print('\n(Step 14 of 14) Calculate goodness of fit.\n')
r2 = reg.score(movie_parameters_np, movie_ratings_np)
print('R-squared value =', r2)

end = time.perf_counter()
print(f'\nProcess done. Program run-time: {end - start:.2f} seconds.')

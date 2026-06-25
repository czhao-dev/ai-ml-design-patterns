#!/bin/bash
set -e

# Movie Recommender Graph Theory - Full Pipeline Runner
#
# Runs the full pipeline against DATA_DIR (default data/sample), writing all
# intermediate and final outputs to RESULTS_DIR (default results). Override
# either by exporting the env var before running, e.g.:
#   DATA_DIR=data/full RESULTS_DIR=results_full ./run_all.sh

export DATA_DIR="${DATA_DIR:-data/sample}"
export RESULTS_DIR="${RESULTS_DIR:-results}"
PYTHON="${PYTHON:-python3}"

echo "========================================="
echo "Movie Recommender Graph Theory - Pipeline"
echo "  DATA_DIR=$DATA_DIR"
echo "  RESULTS_DIR=$RESULTS_DIR"
echo "========================================="

echo ""
echo "[1/10] Building actor/movie dictionaries and actor-actor edgelist..."
$PYTHON src/python/build_actor_movie_dicts.py

echo ""
echo "[2/10] Computing actor PageRank scores..."
$PYTHON src/python/compute_actor_pagerank.py

echo ""
echo "[3/10] Building the movie-movie similarity network..."
$PYTHON src/python/build_movie_network.py

echo ""
echo "[4/10] Detecting communities and analyzing neighbors (pass 1, before genre tagging)..."
$PYTHON src/python/detect_communities_and_neighbors.py

echo ""
echo "[5/10] Mapping movie genres..."
$PYTHON src/python/build_movie_genre_index.py

echo ""
echo "[6/10] Detecting communities and analyzing neighbors (pass 2, with genre tagging)..."
$PYTHON src/python/detect_communities_and_neighbors.py

echo ""
echo "[7/10] Mapping movie ratings..."
$PYTHON src/python/build_movie_rating_index.py

echo ""
echo "[8/10] Predicting ratings via neighborhood averaging..."
$PYTHON src/python/predict_ratings_neighbors.py

echo ""
echo "[9/10] Predicting ratings via linear regression..."
$PYTHON src/python/predict_ratings_regression.py

echo ""
echo "[10/10] Predicting ratings via bipartite actor-movie graph..."
$PYTHON src/python/predict_ratings_bipartite.py

echo ""
echo "========================================="
echo "Pipeline complete. See $RESULTS_DIR/predictions/ for outputs."
echo "========================================="

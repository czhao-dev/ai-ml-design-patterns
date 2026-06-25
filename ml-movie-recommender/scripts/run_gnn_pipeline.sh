#!/bin/bash
set -e

# One-command GNN smoke test against the IMDb sample data: run_all.sh (igraph
# stage) -> export_features_for_gnn.py (bridge) -> train.py -> evaluate.py.
#
# Needs two separate Python environments (see README "Getting Started"):
#   PYTHON_IGRAPH -- python-igraph/scikit-learn venv, e.g. .venv/bin/python3
#   PYTHON_GNN    -- torch/torch-geometric venv, e.g. .venv-gnn/bin/python3
# (PyTorch wheels commonly lag the newest CPython release by months, so the
# GNN venv may need an older Python than whatever the igraph venv uses.)

PYTHON_IGRAPH="${PYTHON_IGRAPH:-.venv/bin/python3}"
PYTHON_GNN="${PYTHON_GNN:-.venv-gnn/bin/python3}"
CONFIG="${CONFIG:-configs/imdb_sample.yaml}"

echo "========================================="
echo "GNN smoke test (IMDb sample)"
echo "  PYTHON_IGRAPH=$PYTHON_IGRAPH"
echo "  PYTHON_GNN=$PYTHON_GNN"
echo "  CONFIG=$CONFIG"
echo "========================================="

echo ""
echo "[1/4] Running the igraph feature-engineering stage (run_all.sh)..."
PYTHON="$PYTHON_IGRAPH" ./run_all.sh

echo ""
echo "[2/4] Exporting flat features for the GNN stage..."
"$PYTHON_IGRAPH" src/python/export_features_for_gnn.py

echo ""
echo "[3/4] Training the GNN (smoke-test pass)..."
"$PYTHON_GNN" scripts/train.py --config "$CONFIG"

echo ""
echo "[4/4] Evaluating (leave-one-out CV on the sample) and writing reports/results_summary.md..."
"$PYTHON_GNN" scripts/evaluate.py --config "$CONFIG"

echo ""
echo "========================================="
echo "Done. See reports/results_summary.md."
echo "========================================="

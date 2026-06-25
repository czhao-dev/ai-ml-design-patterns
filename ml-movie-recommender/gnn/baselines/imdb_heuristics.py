"""Parse the three existing heuristic/linear-regression prediction files.

These are deliberately *parsers*, not re-implementations or subprocess
wrappers around predict_ratings_*.py: those scripts run in the igraph venv
(python-igraph, scikit-learn), which is a different Python install than the
GNN venv (torch, torch-geometric) this module runs in. Rather than invoke
one venv's interpreter from the other, this module just reads the prediction
files run_all.sh / scripts/run_gnn_pipeline.sh have already produced in
RESULTS_DIR/predictions/ -- zero risk of changing the existing scripts'
verified behavior, and no cross-venv subprocess plumbing.
"""

from pathlib import Path


def load_neighbor_predictions(results_dir):
    """{movie_name: rating or None} from neighbor_averaging_predictions.txt."""
    path = Path(results_dir) / "predictions" / "neighbor_averaging_predictions.txt"
    predictions = {}
    with open(path) as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            name_with_id = fields[0]
            name = name_with_id.rsplit(" (id=", 1)[0]
            predictions[name] = None if fields[1] == "NA" else float(fields[1])
    return predictions


def load_regression_predictions(results_dir):
    """{movie_name: rating} from regression_predictions.txt."""
    path = Path(results_dir) / "predictions" / "regression_predictions.txt"
    predictions = {}
    with open(path) as f:
        for line in f:
            name, score = line.rstrip("\n").split("\t")
            predictions[name] = float(score)
    return predictions


def load_bipartite_predictions(results_dir):
    """{movie_name: rating} from bipartite_predictions.txt."""
    path = Path(results_dir) / "predictions" / "bipartite_predictions.txt"
    predictions = {}
    with open(path) as f:
        for line in f:
            name, score = line.rstrip("\n").split("\t")
            predictions[name] = float(score)
    return predictions


def load_all_baselines(results_dir):
    """{movie_name: {"neighbor_averaging": ..., "linear_regression": ..., "bipartite_graph": ...}}."""
    neighbor = load_neighbor_predictions(results_dir)
    regression = load_regression_predictions(results_dir)
    bipartite = load_bipartite_predictions(results_dir)
    names = set(neighbor) | set(regression) | set(bipartite)
    return {
        name: {
            "neighbor_averaging": neighbor.get(name),
            "linear_regression": regression.get(name),
            "bipartite_graph": bipartite.get(name),
        }
        for name in names
    }

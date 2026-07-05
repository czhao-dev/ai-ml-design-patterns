"""Evaluate a GNN track and (re)generate reports/results_summary.md.

IMDb track (configs/imdb_sample.yaml, split_strategy: loo):
    Runs leave-one-out CV from scratch (trains one fresh model per held-out
    movie -- doesn't reuse scripts/train.py's smoke-test checkpoint, since a
    single checkpoint can't represent N different LOO folds). Reports
    aggregate RMSE/MAE across all labeled movies, plus the GNN's prediction
    for the three demo movies next to the three existing heuristic baselines.

IMDb track (configs/imdb_full.yaml, split_strategy: holdout):
    Documented only -- not runnable in this repo (see configs/imdb_full.yaml).

MovieLens track:
    Loads a checkpoint if --checkpoint is given and exists, else trains
    fresh using the config. Reports RMSE/MAE plus Precision/Recall/NDCG@K
    against mean/popularity baselines.

Usage:
    python scripts/evaluate.py --config configs/imdb_sample.yaml
    python scripts/evaluate.py --config configs/movielens_small.yaml [--checkpoint experiments/runs/movielens_small/checkpoints/latest.pt]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnn import training
from gnn.baselines.imdb_heuristics import load_all_baselines
from gnn.baselines.movielens_baselines import (
    GlobalMeanBaseline,
    ItemMeanBaseline,
    PopularityBaseline,
    UserMeanBaseline,
)
from gnn.data.imdb_hetero_data import build_imdb_hetero_data
from gnn.data.movielens_hetero_data import build_movielens_hetero_data
from gnn.data.splits import holdout_label_split, leave_one_out_label_splits
from gnn.metrics.ranking import evaluate_ranking_for_user
from gnn.metrics.regression import mae, rmse
from gnn.utils import get_device, load_config, set_seed

REPORTS_DIR = Path("reports")


def evaluate_imdb(config, device):
    data = build_imdb_hetero_data(config["data"]["export_dir"])
    label_indices = data["movie"].label_mask.nonzero(as_tuple=True)[0].tolist()
    demo_movies = data.demo_movies
    demo_indices = [m["index"] for m in demo_movies]

    epochs = config["training"]["epochs"]
    lr, wd = config["training"]["lr"], config["training"]["weight_decay"]
    nl_cfg = config["training"].get("neighbor_loader")
    nl_kwargs = {"batch_size": nl_cfg["batch_size"], "num_neighbors": nl_cfg["num_neighbors"]} if nl_cfg else None
    if nl_kwargs:
        print(f"Using mini-batch NeighborLoader (batch_size={nl_kwargs['batch_size']}, "
              f"num_neighbors={nl_kwargs['num_neighbors']}).")

    def train_and_predict(train_idx, predict_idx, encoder, decoder):
        if nl_kwargs:
            training.train_imdb_minibatch(data, encoder, decoder, train_idx, epochs, lr, wd, device, **nl_kwargs)
            return training.predict_imdb_minibatch(data, encoder, decoder, predict_idx, device, **nl_kwargs)
        training.train_imdb(data, encoder, decoder, train_idx, epochs, lr, wd, device)
        return training.predict_imdb(data, encoder, decoder, predict_idx, device)

    predictions_by_movie = {}
    demo_in_sample = set()
    global_mean_metrics = None
    split_strategy = config["data"]["split_strategy"]
    if split_strategy == "loo":
        print(f"Running leave-one-out CV across {len(label_indices)} labeled movies...")
        for train_idx, test_idx in leave_one_out_label_splits(label_indices):
            encoder, decoder = training.make_imdb_model(data, config["model"])
            pred = train_and_predict(train_idx, test_idx, encoder, decoder)
            predictions_by_movie[test_idx[0]] = float(pred[0])
        predictions_for_table = predictions_by_movie
    elif split_strategy == "holdout":
        train_idx, val_idx, test_idx = holdout_label_split(
            label_indices, config["data"]["train_frac"], config["data"]["val_frac"], config.get("seed", 0),
        )
        encoder, decoder = training.make_imdb_model(data, config["model"])
        preds = train_and_predict(train_idx, test_idx, encoder, decoder)
        predictions_by_movie = dict(zip(test_idx, preds.tolist()))

        # Demo movies aren't guaranteed to land in the test split at this scale
        # (unlike the sample's LOO, which covers every labeled movie) -- predict
        # them too for the results table, flagging any that ended up in-sample.
        demo_in_sample = {idx for idx in demo_indices if idx in set(train_idx)}
        demo_to_predict = [idx for idx in demo_indices if idx not in predictions_by_movie]
        predictions_for_table = dict(predictions_by_movie)
        if demo_to_predict:
            if nl_kwargs:
                demo_preds = training.predict_imdb_minibatch(data, encoder, decoder, demo_to_predict, device, **nl_kwargs)
            else:
                demo_preds = training.predict_imdb(data, encoder, decoder, demo_to_predict, device)
            predictions_for_table.update(zip(demo_to_predict, demo_preds.tolist()))

        # A trivial baseline (predict the train-label mean for every test movie)
        # is the minimum bar a full-scale GNN needs to clear -- at this scale,
        # unlike the 7-movie sample, there's enough data for this comparison to
        # be meaningful rather than noise.
        train_mean = float(data["movie"].y[train_idx].mean())
        test_actual = [float(data["movie"].y[i]) for i in test_idx]
        global_mean_pred = [train_mean] * len(test_idx)
        global_mean_metrics = (rmse(global_mean_pred, test_actual), mae(global_mean_pred, test_actual))
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}")

    actual = data["movie"].y
    preds_list = [predictions_by_movie[i] for i in sorted(predictions_by_movie)]
    actual_list = [float(actual[i]) for i in sorted(predictions_by_movie)]
    gnn_rmse, gnn_mae = rmse(preds_list, actual_list), mae(preds_list, actual_list)
    print(f"GNN ({split_strategy}) RMSE={gnn_rmse:.4f} MAE={gnn_mae:.4f} "
          f"over {len(preds_list)} labeled movies.")

    results_dir = Path(config["data"]["export_dir"]).parent
    baselines = load_all_baselines(results_dir)
    write_imdb_report(demo_movies, predictions_for_table, actual, baselines, gnn_rmse, gnn_mae, len(preds_list),
                       split_strategy, demo_in_sample, global_mean_metrics)


def write_imdb_report(demo_movies, predictions_by_movie, actual, baselines, gnn_rmse, gnn_mae, n_eval,
                       split_strategy="loo", demo_in_sample=frozenset(), global_mean_metrics=None):
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / "results_summary.md"
    data_label = "sample data" if split_strategy == "loo" else f"full IMDb data, {n_eval:,} labeled movies"
    lines = ["# Results Summary\n", f"\n## IMDb track (rating prediction, {data_label})\n"]
    lines.append(
        "\n| Method | " + " | ".join(m["name"].split(" (")[0] for m in demo_movies) + " |\n"
    )
    lines.append("|---" * (len(demo_movies) + 1) + "|\n")

    def row(label, values):
        return "| " + label + " | " + " | ".join(f"{v:.2f}" if v is not None else "NA" for v in values) + " |\n"

    name_lookup = {m["index"]: m["name"] for m in demo_movies}
    for key, label in [("neighbor_averaging", "Neighborhood Averaging"),
                        ("linear_regression", "Linear Regression"),
                        ("bipartite_graph", "Bipartite Graph Averaging")]:
        values = [baselines[name_lookup[m["index"]]][key] for m in demo_movies]
        lines.append(row(label, values))
    gnn_values = [predictions_by_movie.get(m["index"]) for m in demo_movies]
    has_in_sample_demo = any(m["index"] in demo_in_sample for m in demo_movies)
    gnn_label = f"GNN (heterogeneous, {split_strategy.upper()})" + (" [*]" if has_in_sample_demo else "")
    lines.append(row(gnn_label, gnn_values))
    actual_values = [float(actual[m["index"]]) for m in demo_movies]
    lines.append(row(f"**IMDb ({data_label})**", actual_values))
    if has_in_sample_demo:
        in_sample_names = [m["name"] for m in demo_movies if m["index"] in demo_in_sample]
        lines.append(
            f"\n[*] {', '.join(in_sample_names)}: in-sample prediction(s) -- this movie landed in the "
            "training split (holdout is a random split, so this can happen by chance), not held out, "
            "so its GNN value above is not comparable to the others.\n"
        )

    lines.append(
        f"\nGNN {split_strategy.upper()} across all {n_eval:,} labeled movies: "
        f"RMSE={gnn_rmse:.4f}, MAE={gnn_mae:.4f}. "
        "The three heuristic baselines only ever produced predictions for the three demo "
        "movies shown above (that's all the original pipeline computed), so no comparable "
        f"full-sample RMSE/MAE exists for them -- only the GNN's {split_strategy.upper()} metric covers "
        "all labeled movies.\n"
    )
    if global_mean_metrics is not None:
        gm_rmse, gm_mae = global_mean_metrics
        verdict = "beats" if gnn_rmse < gm_rmse else "does not beat"
        lines.append(
            f"\n**Global Mean baseline** (predict the train-label average for every test movie): "
            f"RMSE={gm_rmse:.4f}, MAE={gm_mae:.4f} (same test split as the GNN's {split_strategy.upper()} metric above).\n"
        )
        lines.append(
            f"\n**The GNN {verdict} this trivial baseline** on this run (GNN RMSE={gnn_rmse:.4f} vs. "
            f"Global Mean RMSE={gm_rmse:.4f}). This is consistent with the very weak signal the full-scale "
            "linear-regression heuristic already found in this data (R^2 ~= 0.02 on all labeled movies, vs. "
            "R^2 ~= 0.42 on the tiny, unrepresentative 7-movie sample -- see the sample-scale note above): "
            "actor PageRank/degree/cast-structure/community/genre features alone carry little signal about "
            "a movie's aggregate rating at real scale. Plausible next steps to actually close this gap -- not "
            "attempted here -- include more training epochs / wider neighbor sampling (this run used 10 "
            "epochs, `num_neighbors: [15, 10]`, `hidden_dim: 64`), additional node features (e.g. runtime, "
            "release year, cast size beyond a single log-count), or accepting that rating regression from "
            "cast/graph structure alone is a genuinely hard problem at this scale.\n"
        )
    if split_strategy == "loo":
        lines.append(
            "\n**Methodology note**: every number above is an out-of-sample prediction "
            "(leave-one-out cross-validation -- each movie's prediction comes from a model that "
            "never saw that movie's rating during training). On a sample this small (7 labeled "
            "movies), a strong-looking RMSE only shows the pipeline runs correctly end-to-end, "
            "not that the architecture works at scale -- a GNN can memorize 7 points as easily as "
            "the existing LinearRegression baseline overfits them on this same sample "
            "(R^2 ~= 0.42 on this sample vs the full-scale numbers below).\n"
        )
    else:
        lines.append(
            "\n**Methodology note**: RMSE/MAE above are computed only from a genuinely held-out "
            f"random test split ({n_eval:,} movies never seen during training); any in-sample "
            "demo-movie predictions flagged above are excluded from these metrics. Unlike the tiny "
            "sample, this is real evidence of (or against) the architecture generalizing, not just "
            "proof the pipeline runs.\n"
        )
    path.write_text("".join(lines))
    print(f"Wrote {path}")


def evaluate_movielens(config, device, checkpoint_path):
    data = build_movielens_hetero_data(
        config["data"]["data_dir"],
        split_strategy=config["data"]["split_strategy"],
        min_ratings_per_user=config["data"]["min_ratings_per_user"],
        seed=config.get("seed", 0),
    )
    encoder, decoder = training.make_movielens_model(data, config["model"])
    encoder.to(device)
    decoder.to(device)

    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        encoder.load_state_dict({k[len("encoder."):]: v for k, v in state["model"].items() if k.startswith("encoder.")})
        decoder.load_state_dict({k[len("decoder."):]: v for k, v in state["model"].items() if k.startswith("decoder.")})
        encoder.to(device)
        decoder.to(device)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint given/found -- training fresh.")
        training.train_movielens(
            data, encoder, decoder, config["training"]["epochs"],
            config["training"]["lr"], config["training"]["weight_decay"], device,
        )

    rel = data["user", "rates", "movie"]
    gnn_test_pred = training.predict_movielens(data, encoder, decoder, rel.test_edge_label_index, device)
    gnn_rmse = rmse(gnn_test_pred.tolist(), rel.test_edge_label.tolist())
    gnn_mae = mae(gnn_test_pred.tolist(), rel.test_edge_label.tolist())

    global_baseline = GlobalMeanBaseline()
    global_baseline.fit(rel.train_edge_label)
    user_baseline = UserMeanBaseline()
    user_baseline.fit(rel.train_edge_label_index, rel.train_edge_label, data["user"].num_nodes)
    item_baseline = ItemMeanBaseline()
    item_baseline.fit(rel.train_edge_label_index, rel.train_edge_label, data["movie"].num_nodes)
    popularity = PopularityBaseline()
    popularity.fit(rel.train_edge_label_index, data["movie"].num_nodes)

    baseline_metrics = {}
    for name, baseline in [("Global Mean", global_baseline), ("User Mean", user_baseline), ("Item Mean", item_baseline)]:
        pred = baseline.predict(rel.test_edge_label_index)
        baseline_metrics[name] = (rmse(pred.tolist(), rel.test_edge_label.tolist()),
                                   mae(pred.tolist(), rel.test_edge_label.tolist()))

    k_values = config["eval"]["k_values"]
    threshold = config["eval"]["relevance_threshold"]
    test_users = sorted(set(rel.test_edge_label_index[0].tolist()))
    train_movies_by_user = data.eval_meta["train_movies_by_user"]

    gnn_scores = training.score_all_movies_for_users(data, encoder, decoder, device, test_users)
    pop_scores = popularity.score_all_movies()

    gnn_agg = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_values}
    pop_agg = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_values}
    test_idx_by_user = {}
    for col, u in enumerate(rel.test_edge_label_index[0].tolist()):
        test_idx_by_user.setdefault(u, []).append(col)

    for u in test_users:
        seen = train_movies_by_user.get(u, set())
        cols = test_idx_by_user[u]
        relevant = {
            int(rel.test_edge_label_index[1, c]) for c in cols if float(rel.test_edge_label[c]) >= threshold
        }
        candidates = [m for m in range(data["movie"].num_nodes) if m not in seen]

        gnn_scored = [(m, float(gnn_scores[u][m])) for m in candidates]
        pop_scored = [(m, float(pop_scores[m])) for m in candidates]
        gnn_result = evaluate_ranking_for_user(gnn_scored, relevant, k_values)
        pop_result = evaluate_ranking_for_user(pop_scored, relevant, k_values)
        for k in k_values:
            for metric in ("precision", "recall", "ndcg"):
                gnn_agg[k][metric].append(gnn_result[k][metric])
                pop_agg[k][metric].append(pop_result[k][metric])

    def avg(agg, k, metric):
        values = agg[k][metric]
        return sum(values) / len(values) if values else 0.0

    write_movielens_report(gnn_rmse, gnn_mae, baseline_metrics, gnn_agg, pop_agg, k_values, avg, len(test_users))


def write_movielens_report(gnn_rmse, gnn_mae, baseline_metrics, gnn_agg, pop_agg, k_values, avg, n_users):
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / "results_summary.md"
    existing = path.read_text() if path.exists() else "# Results Summary\n"
    if "## MovieLens track" in existing:
        existing = existing.split("## MovieLens track")[0]
    existing = existing.rstrip("\n") + "\n"

    lines = [existing, "\n## MovieLens track (personalized recommendation, ml-latest-small)\n"]
    lines.append("\n| Method | Test RMSE | Test MAE |\n|---|---|---|\n")
    lines.append(f"| GNN (heterogeneous) | {gnn_rmse:.4f} | {gnn_mae:.4f} |\n")
    for name, (r, m) in baseline_metrics.items():
        lines.append(f"| {name} | {r:.4f} | {m:.4f} |\n")

    lines.append("\nRanking (full-catalog, not sampled negatives -- see gnn/metrics/ranking.py):\n")
    header = "| Method | " + " | ".join(f"P@{k} / R@{k} / NDCG@{k}" for k in k_values) + " |\n"
    lines.append("\n" + header)
    lines.append("|---" * (len(k_values) + 1) + "|\n")
    gnn_row = "| GNN | " + " | ".join(
        f"{avg(gnn_agg, k, 'precision'):.3f} / {avg(gnn_agg, k, 'recall'):.3f} / {avg(gnn_agg, k, 'ndcg'):.3f}"
        for k in k_values
    ) + " |\n"
    pop_row = "| Popularity (non-personalized) | " + " | ".join(
        f"{avg(pop_agg, k, 'precision'):.3f} / {avg(pop_agg, k, 'recall'):.3f} / {avg(pop_agg, k, 'ndcg'):.3f}"
        for k in k_values
    ) + " |\n"
    lines.append(gnn_row)
    lines.append(pop_row)
    lines.append(
        f"\nEvaluated over {n_users} users with a held-out test rating "
        "(leave-one-out per user; users with fewer than 3 ratings have no held-out "
        "test/val rating and are excluded from this table). Candidates for ranking exclude "
        "movies the user already rated in train.\n"
    )
    lines.append(
        "\n**Note**: a GNN trained with plain MSE rating-regression loss optimizes for "
        "predicting *how much* a user would rate a movie, which is not the same objective as "
        "ranking the single correct held-out movie above ~9,700 candidates -- a well-documented "
        "mismatch in the recommender-systems literature (e.g. Cremonesi et al. 2010, "
        "\"Performance of Recommender Algorithms on Top-N Recommendation Tasks\"). A ranking-aware "
        "training objective (e.g. BPR loss over sampled negatives) is the natural next step if "
        "top-N ranking quality matters more than rating RMSE -- not implemented here.\n"
    )
    path.write_text("".join(lines))
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 0))
    device = get_device(args.device or config["training"].get("device"))

    if config["data"]["track"] == "imdb":
        evaluate_imdb(config, device)
    elif config["data"]["track"] == "movielens":
        evaluate_movielens(config, device, args.checkpoint)
    else:
        raise ValueError(f"Unknown data.track: {config['data']['track']}")


if __name__ == "__main__":
    main()

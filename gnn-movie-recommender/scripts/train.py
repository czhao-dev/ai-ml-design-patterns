"""Single training entry point for both GNN tracks.

Usage:
    python scripts/train.py --config configs/imdb_sample.yaml
    python scripts/train.py --config configs/movielens_small.yaml [--epochs 5]

The config's `task` field (node_regression vs edge_regression) and
`data.track` field (imdb vs movielens) select which HeteroData builder,
model, and training loop to use -- see gnn/training.py.

For the IMDb sample config (split_strategy: loo), this script trains a
single smoke-test model on *all* labeled movies, mainly to verify the
pipeline runs end-to-end and to produce a checkpoint -- the actual reported
metric (leave-one-out CV RMSE/MAE) is computed independently by
scripts/evaluate.py, which retrains a fresh model per held-out movie.
"""

import argparse
import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnn import training
from gnn.data.imdb_hetero_data import build_imdb_hetero_data
from gnn.data.movielens_hetero_data import build_movielens_hetero_data
from gnn.utils import get_device, load_config, save_checkpoint, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs (for smoke tests).")
    parser.add_argument("--device", default=None, help="Override training.device.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 0))
    device = get_device(args.device or config["training"].get("device"))
    epochs = args.epochs or config["training"]["epochs"]
    track = config["data"]["track"]

    if track == "imdb":
        data = build_imdb_hetero_data(config["data"]["export_dir"])
        encoder, decoder = training.make_imdb_model(data, config["model"])
        label_indices = data["movie"].label_mask.nonzero(as_tuple=True)[0].tolist()
        print(f"IMDb track: {len(label_indices)} labeled movies, training on all of them "
              f"(smoke test; see scripts/evaluate.py for the real LOO-CV/holdout metric).")
        nl_cfg = config["training"].get("neighbor_loader")
        if nl_cfg:
            print(f"Using mini-batch NeighborLoader training (batch_size={nl_cfg['batch_size']}, "
                  f"num_neighbors={nl_cfg['num_neighbors']}).")
            loss = training.train_imdb_minibatch(
                data, encoder, decoder, label_indices, epochs,
                config["training"]["lr"], config["training"]["weight_decay"], device,
                batch_size=nl_cfg["batch_size"], num_neighbors=nl_cfg["num_neighbors"],
            )
        else:
            loss = training.train_imdb(
                data, encoder, decoder, label_indices, epochs,
                config["training"]["lr"], config["training"]["weight_decay"], device,
            )
        print(f"Final train MSE: {loss:.4f}")
    elif track == "movielens":
        data = build_movielens_hetero_data(
            config["data"]["data_dir"],
            split_strategy=config["data"]["split_strategy"],
            min_ratings_per_user=config["data"]["min_ratings_per_user"],
            seed=config.get("seed", 0),
        )
        encoder, decoder = training.make_movielens_model(data, config["model"])
        print(f"MovieLens track: {data['user'].num_nodes} users, {data['movie'].num_nodes} movies, "
              f"{data['user', 'rates', 'movie'].train_edge_label_index.size(1)} train ratings.")
        loss = training.train_movielens(
            data, encoder, decoder, epochs,
            config["training"]["lr"], config["training"]["weight_decay"], device,
        )
        print(f"Final train MSE: {loss:.4f}")
    else:
        raise ValueError(f"Unknown data.track: {track}")

    combined = nn.ModuleDict({"encoder": encoder, "decoder": decoder})
    checkpoint_path = Path(config["output"]["run_dir"]) / "checkpoints" / "latest.pt"
    save_checkpoint(checkpoint_path, combined, extra={"config": config})
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()

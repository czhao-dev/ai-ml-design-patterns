"""Model construction + training loops shared by scripts/train.py and scripts/evaluate.py.

Kept deliberately thin: build the encoder/decoder for a track, then run a
plain full-batch training loop (both datasets here are small enough that
full-batch training is the right call -- mini-batch neighbor sampling is a
documented follow-up for the full IMDb scale, not implemented here, see
configs/imdb_full.yaml).
"""

import torch
import torch.nn.functional as F

from gnn.models.hetero_gnn import EdgeRatingDecoder, HeteroGNNEncoder, RatingDecoder


def make_imdb_model(data, model_cfg):
    edge_types = list(data.edge_types)
    encoder = HeteroGNNEncoder(
        edge_types=edge_types,
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg.get("dropout", 0.1),
        conv_type=model_cfg.get("conv_type", "sage"),
        node_feature_dims={"actor": data["actor"].x.size(1), "movie": data["movie"].x.size(1)},
    )
    decoder = RatingDecoder(model_cfg["hidden_dim"])
    return encoder, decoder


def make_movielens_model(data, model_cfg):
    edge_types = list(data.edge_types)
    encoder = HeteroGNNEncoder(
        edge_types=edge_types,
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg.get("dropout", 0.1),
        conv_type=model_cfg.get("conv_type", "sage"),
        node_feature_dims={"movie": data["movie"].x.size(1)},
        node_embedding_counts={"user": data["user"].num_nodes},
    )
    decoder = EdgeRatingDecoder(model_cfg["hidden_dim"])
    return encoder, decoder


def _imdb_forward(data, encoder, decoder, device):
    x_dict = {"actor": data["actor"].x.to(device), "movie": data["movie"].x.to(device)}
    edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}
    edge_weight_dict = {
        et: data[et].edge_weight.to(device) for et in data.edge_types if "edge_weight" in data[et]
    }
    h_dict = encoder(x_dict, edge_index_dict, edge_weight_dict)
    return decoder(h_dict["movie"])


def train_imdb(data, encoder, decoder, train_idx, epochs, lr, weight_decay, device):
    """Trains in place on the given labeled-movie indices; returns the final train loss."""
    encoder.to(device)
    decoder.to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    y = data["movie"].y.to(device)

    encoder.train()
    decoder.train()
    loss = None
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = _imdb_forward(data, encoder, decoder, device)
        loss = F.mse_loss(pred[train_idx_t], y[train_idx_t])
        loss.backward()
        optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def predict_imdb(data, encoder, decoder, movie_indices, device):
    encoder.eval()
    decoder.eval()
    pred = _imdb_forward(data, encoder, decoder, device)
    idx_t = torch.tensor(movie_indices, dtype=torch.long, device=device)
    return pred[idx_t].cpu()


def _movielens_forward(data, encoder, decoder, device, edge_label_index):
    x_dict = {
        "user": torch.arange(data["user"].num_nodes, device=device),
        "movie": data["movie"].x.to(device),
    }
    edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}
    h_dict = encoder(x_dict, edge_index_dict)
    return decoder(h_dict["user"], h_dict["movie"], edge_label_index.to(device))


def train_movielens(data, encoder, decoder, epochs, lr, weight_decay, device):
    encoder.to(device)
    decoder.to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    rel = data["user", "rates", "movie"]
    train_label = rel.train_edge_label.to(device)

    encoder.train()
    decoder.train()
    loss = None
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = _movielens_forward(data, encoder, decoder, device, rel.train_edge_label_index)
        loss = F.mse_loss(pred, train_label)
        loss.backward()
        optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def predict_movielens(data, encoder, decoder, edge_label_index, device):
    encoder.eval()
    decoder.eval()
    return _movielens_forward(data, encoder, decoder, device, edge_label_index).cpu()


@torch.no_grad()
def score_all_movies_for_users(data, encoder, decoder, device, user_ids):
    """Scores every movie for each user in user_ids -- used for top-N ranking eval."""
    encoder.eval()
    decoder.eval()
    num_movies = data["movie"].num_nodes
    x_dict = {
        "user": torch.arange(data["user"].num_nodes, device=device),
        "movie": data["movie"].x.to(device),
    }
    edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}
    h_dict = encoder(x_dict, edge_index_dict)

    scores_by_user = {}
    for user_id in user_ids:
        movie_idx = torch.arange(num_movies, device=device)
        user_idx = torch.full((num_movies,), user_id, device=device)
        edge_label_index = torch.stack([user_idx, movie_idx])
        scores_by_user[user_id] = decoder(h_dict["user"], h_dict["movie"], edge_label_index).cpu()
    return scores_by_user

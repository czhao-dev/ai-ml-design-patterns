"""Shared heterogeneous GNN encoder plus track-specific decoder heads.

HeteroGNNEncoder is shared between the IMDb (node regression) and MovieLens
(edge regression / ranking) tracks. The two decoder heads are kept separate
since a node-level head and an edge-level head have genuinely different
input shapes -- forcing a common base class would just be indirection.

Edge types that carry an edge_weight (co_appeared_with, similar_to) use
GraphConv, which natively supports weighted message passing; unweighted
bipartite edge types (acted_in/performed_by, rates/rated_by) use SAGEConv.
GraphConv/SAGEConv (rather than GATConv) are the defaults because they are
robust on graphs with many degree-1/2 nodes -- exactly what the tiny IMDb
sample looks like -- where attention-softmax over 1-2 neighbors adds
parameters without adding signal. GATConv is still swappable via config for
larger graphs (see configs/imdb_full.yaml) where attention has enough
neighborhood diversity to be worth it.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GraphConv, HeteroConv, SAGEConv

from gnn.models.layers import mlp_head

WEIGHTED_RELATIONS = {"co_appeared_with", "similar_to"}


def _make_conv(conv_type, hidden_dim, weighted):
    if weighted:
        # GATConv/SAGEConv don't accept edge_weight; GraphConv does.
        return GraphConv(hidden_dim, hidden_dim, aggr="mean")
    if conv_type == "gat":
        return GATConv(hidden_dim, hidden_dim, heads=1, add_self_loops=False)
    return SAGEConv(hidden_dim, hidden_dim)


class HeteroGNNEncoder(nn.Module):
    """Encodes a HeteroData graph into per-node-type hidden embeddings.

    node_feature_dims: {node_type: input feature dim} for node types with
        static feature matrices (e.g. "movie", and "actor").
    node_embedding_counts: {node_type: num_nodes} for node types with no
        side features (e.g. MovieLens "user"), encoded via a learned
        nn.Embedding instead of a Linear projection.
    """

    def __init__(self, edge_types, hidden_dim, num_layers=2, dropout=0.1,
                 conv_type="sage", node_feature_dims=None, node_embedding_counts=None):
        super().__init__()
        node_feature_dims = node_feature_dims or {}
        node_embedding_counts = node_embedding_counts or {}

        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(dim, hidden_dim) for ntype, dim in node_feature_dims.items()
        })
        self.input_emb = nn.ModuleDict({
            ntype: nn.Embedding(count, hidden_dim) for ntype, count in node_embedding_counts.items()
        })

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            convs = {
                et: _make_conv(conv_type, hidden_dim, et[1] in WEIGHTED_RELATIONS)
                for et in edge_types
            }
            self.layers.append(HeteroConv(convs, aggr="sum"))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        h_dict = {}
        for ntype, proj in self.input_proj.items():
            h_dict[ntype] = proj(x_dict[ntype])
        for ntype, emb in self.input_emb.items():
            h_dict[ntype] = emb(x_dict[ntype])

        for layer in self.layers:
            kwargs = {"edge_weight_dict": edge_weight_dict} if edge_weight_dict else {}
            h_dict = layer(h_dict, edge_index_dict, **kwargs)
            h_dict = {ntype: self.dropout(torch.relu(h)) for ntype, h in h_dict.items()}
        return h_dict


class RatingDecoder(nn.Module):
    """Node-level regression head (IMDb track: predict a movie's rating)."""

    def __init__(self, hidden_dim, mlp_hidden_dim=None):
        super().__init__()
        self.mlp = mlp_head(hidden_dim, mlp_hidden_dim or hidden_dim, out_dim=1)

    def forward(self, node_embeddings):
        return self.mlp(node_embeddings).squeeze(-1)


class EdgeRatingDecoder(nn.Module):
    """Edge-level regression/ranking head (MovieLens track: score a (user, movie) pair)."""

    def __init__(self, hidden_dim, mlp_hidden_dim=None):
        super().__init__()
        self.mlp = mlp_head(2 * hidden_dim, mlp_hidden_dim or hidden_dim, out_dim=1)

    def forward(self, user_embeddings, movie_embeddings, edge_label_index):
        src, dst = edge_label_index
        pair = torch.cat([user_embeddings[src], movie_embeddings[dst]], dim=-1)
        return self.mlp(pair).squeeze(-1)

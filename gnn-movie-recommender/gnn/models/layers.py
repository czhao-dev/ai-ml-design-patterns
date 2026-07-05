"""Small shared building blocks used by both decoder heads."""

import torch.nn as nn


def mlp_head(in_dim, hidden_dim, out_dim=1, dropout=0.0):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )

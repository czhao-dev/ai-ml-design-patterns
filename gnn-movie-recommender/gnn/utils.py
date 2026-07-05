"""Shared config/seed/device/checkpoint helpers for both GNN tracks."""

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path, model, optimizer=None, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state

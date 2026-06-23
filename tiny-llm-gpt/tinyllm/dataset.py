import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    """Wraps a flat uint16 token memmap and yields shifted (x, y) windows."""

    def __init__(self, bin_path: str, context_length: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.data) - self.context_length

    def __getitem__(self, idx: int):
        ctx = self.context_length
        x = torch.from_numpy(self.data[idx:idx + ctx].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1:idx + ctx + 1].astype(np.int64))
        return x, y


def get_batch(bin_path_or_dataset, batch_size: int, context_length: int, device: torch.device):
    """nanoGPT-style random-offset batch sampler directly from a memmap."""
    if isinstance(bin_path_or_dataset, TokenDataset):
        data = bin_path_or_dataset.data
    else:
        data = np.memmap(bin_path_or_dataset, dtype=np.uint16, mode="r")
    idxs = np.random.randint(0, len(data) - context_length, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i:i + context_length].astype(np.int64)) for i in idxs])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + context_length].astype(np.int64)) for i in idxs])
    x, y = x.to(device), y.to(device)
    return x, y

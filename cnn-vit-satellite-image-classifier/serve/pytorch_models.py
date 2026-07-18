"""PyTorch model definitions for the satellite image classifier.

Two architectures are defined here:

  SatelliteCNN   — the standalone CNN trained in script 05, saved as an
                   nn.Sequential state dict (keys are plain integers).

  CNN_ViT_Hybrid — the CNN-ViT hybrid trained in script 08, whose CNN
                   backbone weights are stored under the "cnn.features.*"
                   key prefix (stripped during loading).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Standalone CNN  (matches the nn.Sequential in script 05)
# ---------------------------------------------------------------------------

def build_satellite_cnn(num_classes: int = 2) -> nn.Sequential:
    """Build the CNN exactly as defined in script 05.

    Architecture: 6 conv blocks (3→32→64→128→256→512→1024 channels,
    5×5 kernels, same padding, MaxPool2d(2) + BatchNorm2d after each),
    followed by AdaptiveAvgPool2d(1), Flatten, and two linear layers.

    Returns an nn.Sequential so the saved state-dict keys (plain integers)
    match without any remapping.
    """
    return nn.Sequential(
        # Conv block 1
        nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(32),
        # Conv block 2
        nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(64),
        # Conv block 3
        nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(128),
        # Conv block 4
        nn.Conv2d(128, 256, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(256),
        # Conv block 5
        nn.Conv2d(256, 512, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(512),
        # Conv block 6
        nn.Conv2d(512, 1024, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(1024),
        # Classifier head
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(1024, 2048), nn.ReLU(), nn.BatchNorm1d(2048), nn.Dropout(0.4),
        nn.Linear(2048, num_classes),
    )


# ---------------------------------------------------------------------------
# CNN-ViT hybrid  (matches script 08)
# ---------------------------------------------------------------------------

class _ConvBackbone(nn.Module):
    """CNN feature extractor used as the ViT's patch source (no classifier head)."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(1024),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)  # (B, 1024, H', W')


class _PatchEmbed(nn.Module):
    """Project CNN feature map channels to ViT embedding dimension via 1×1 conv."""

    def __init__(self, in_channels: int = 1024, embed_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, L, D)


class _MHSA(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(B, N, self.heads, -1).transpose(1, 2)
        k = k.reshape(B, N, self.heads, -1).transpose(1, 2)
        v = v.reshape(B, N, self.heads, -1).transpose(1, 2)
        attn = self.attn_drop(
            (torch.matmul(q, k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        )
        return self.proj_drop(self.proj(torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)))


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _MHSA(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _ViT(nn.Module):
    """Vision Transformer that operates on CNN feature-map tokens."""

    def __init__(
        self,
        in_channels: int = 1024,
        num_classes: int = 2,
        embed_dim: int = 768,
        depth: int = 3,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_tokens: int = 50,
    ) -> None:
        super().__init__()
        self.patch = _PatchEmbed(in_channels, embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.randn(1, max_tokens, embed_dim))
        self.blocks = nn.ModuleList([
            _TransformerBlock(embed_dim, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x)                          # (B, L, D)
        B, L, _ = x.shape
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)             # (B, L+1, D)
        x = x + self.pos[:, : L + 1]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])        # CLS token → logits


class CNN_ViT_Hybrid(nn.Module):
    """End-to-end CNN-ViT hybrid classifier."""

    def __init__(self, num_classes: int = 2, embed_dim: int = 768, depth: int = 3, heads: int = 6) -> None:
        super().__init__()
        self.cnn = _ConvBackbone()
        self.vit = _ViT(num_classes=num_classes, embed_dim=embed_dim, depth=depth, heads=heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(self.cnn.forward_features(x))

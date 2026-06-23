import torch
import torch.nn as nn

from tinyllm.attention import CausalSelfAttention


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-norm block: x + attn(ln1(x)); x + ffn(ln2(x))."""

    def __init__(self, hidden_size: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, n_heads, dropout)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyllm.transformer import TransformerBlock


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        n_layers: int,
        n_heads: int,
        hidden_size: int,
        dropout: float,
    ):
        super().__init__()
        self.context_length = context_length
        self.n_layers = n_layers

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(context_length, hidden_size)
        self.embed_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            TransformerBlock(hidden_size, n_heads, dropout) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Weight tying: LM head shares weights with the token embedding.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.context_length, "sequence length exceeds context_length"

        pos = torch.arange(0, T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.embed_dropout(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

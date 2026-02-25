from __future__ import annotations

from typing import TYPE_CHECKING

import pytorch_lightning as L
import torch
from torch import nn
from torch.nn import functional as F

from .positional import RotaryPositionalEmbeddings

if TYPE_CHECKING:
    from .config import Config


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.embed_dims = config.embed_dims
        self.head_dims = self.embed_dims // self.n_heads
        self.dropout = config.dropout
        if self.embed_dims % self.n_heads != 0:
            raise ValueError(f"dims {self.embed_dims} is not evenly divisible by heads {self.n_heads}")
        
        self.pos_embed = RotaryPositionalEmbeddings(self.head_dims, config.rotary_base, config.max_len)

        self.q_proj = nn.Linear(self.embed_dims, self.n_heads * self.head_dims, bias=False)
        self.k_proj = nn.Linear(self.embed_dims, self.n_heads * self.head_dims, bias=False)
        self.v_proj = nn.Linear(self.embed_dims, self.n_heads * self.head_dims, bias=False)
        self.o_proj = nn.Linear(self.embed_dims, self.embed_dims, bias=False)
        self.ff_up = nn.Linear(config.embed_dims, config.embed_dims * 4, bias=False)
        self.ff_down = nn.Linear(config.embed_dims * 4, config.embed_dims, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        x_norm = norm(x)

        q = self.q_proj(x_norm).view(B, self.n_heads, S, self.head_dims).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, self.n_heads, S, self.head_dims).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, self.n_heads, S, self.head_dims).transpose(1, 2)

        q, k = self.pos_embed(q, k)

        output = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout if self.training else 0.0, 
            is_causal=True
        )

        output = output.transpose(1, 2).reshape(B, S, D)
        output = self.o_proj(output)

        x = x + output
        ff_out = self.ff_down(F.gelu(self.ff_up(norm(x))))
        x = x + ff_out

        return x

class ChessFormer(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.embed = nn.Embedding(config.vocab_size, config.embed_dims)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.layers)])
        self.lm_head = nn.Linear(config.embed_dims, config.vocab_size, bias=False)

        self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)

        x = norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0  # Pad is 0, exclude
            )
        return logits, loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, targets = batch[:, :-1], batch[:, 1:]
        _, loss = self(x, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss  # type: ignore

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, targets = batch[:, :-1], batch[:, 1:]
        _, loss = self(x, targets)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss  # type: ignore
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.wd
        )
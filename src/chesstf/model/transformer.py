from __future__ import annotations

from typing import TYPE_CHECKING

import pytorch_lightning as L
import torch
from torch import nn
from torch.nn import functional as F

from chesstf.model.legality_metric import LegalityMetric
from chesstf.model.positional import RotaryPositionalEmbeddings
from chesstf.model.stockfish_metric import StockfishMetric

if TYPE_CHECKING:
    from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig

    from .config import Config


def norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


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
    def __init__(self, config: Config, id_to_move: dict[int, str], stockfish_path: str | None = None):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=["id_to_move"])

        self.embed = nn.Embedding(config.vocab_size, config.embed_dims)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.layers)])
        self.lm_head = nn.Linear(config.embed_dims, config.vocab_size, bias=False)

        self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

        self.legality_metric = LegalityMetric(id_to_move)
        self.stockfish_metric = StockfishMetric(id_to_move, engine_path=stockfish_path)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert x.min() >= 0 and x.max() < self.config.vocab_size, \
            f"Token ID out of range: {x.min().item() = }, {x.max().item() = }"
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
                ignore_index=-100
            )
        return logits, loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, targets = batch['input_ids'], batch['labels']
        _, loss = self(x, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss  # type: ignore

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, targets = batch['input_ids'], batch['labels']
        logits, loss = self(x, targets)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.legality_metric.update(logits, x)
        self.stockfish_metric.update(logits, x)
        return loss  # type: ignore

    def on_validation_epoch_end(self) -> None:
        self.log("val/legality", self.legality_metric.compute())
        self.log("val/stockfish_cp_loss", self.stockfish_metric.compute())
        self.legality_metric.reset()
        self.stockfish_metric.reset()

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.wd
        )

        def lr_lambda(step: int) -> float:
            warmup = self.config.warmup_steps
            if step < warmup:
                return step / max(1, warmup)
            total = self.trainer.estimated_stepping_batches
            decay_steps = total - warmup
            progress = (step - warmup) / max(1, decay_steps)
            import math
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_ratio = self.config.min_lr / self.config.lr
            return min_ratio + (1.0 - min_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
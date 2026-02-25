from __future__ import annotations

import torch
from torch import nn


class RotaryPositionalEmbeddings(nn.Module):

    cos_table: torch.Tensor
    sin_table: torch.Tensor

    def __init__(self, dim: int, base: int = 10000, max_seq_len: int = 256):
        super().__init__()
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        t = torch.arange(max_seq_len).float()

        freqs = torch.outer(t, inv_freq)

        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_table", emb.cos())
        self.register_buffer("sin_table", emb.sin())

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError(f"x dim -1 must be even but is {x.shape[-1]}")
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]

        if seq_len > self.cos_table.shape[0]:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.cos_table.shape[0]}")

        cos = self.cos_table[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_table[:seq_len].unsqueeze(0).unsqueeze(0)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot
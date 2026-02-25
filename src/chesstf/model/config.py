from dataclasses import dataclass

@dataclass
class Config:
    layers: int = 8
    n_heads: int = 4
    embed_dims: int = 256
    max_len: int = 256
    dropout: float = 0.1
    lr: float = 1e-5
    wd: float = 0.1
    rotary_base: int = 10000
    vocab_size: int = 1974
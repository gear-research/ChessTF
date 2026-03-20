"""W&B sweep training script.

Usage:
    # 1. Create the sweep (once):
    wandb sweep configs/sweep.yaml

    # 2. Launch an agent (run as many as you have GPUs):
    wandb agent <your-entity>/chesstf/<sweep-id>

    # Or via docker compose (set WANDB_SWEEP_ID in .env):
    docker compose --profile sweep up
"""

from __future__ import annotations

import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger

import wandb
from chesstf.data.tokenizer import SPECIAL_TOKENS, ChessTokenizer
from chesstf.model.config import Config
from chesstf.model.transformer import ChessFormer
from chesstf.training.datamodule import ChessDataModule

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
_SPECIAL_IDS: frozenset[int] = frozenset(SPECIAL_TOKENS.values())


def sweep_train() -> None:
    run = wandb.init()
    assert run is not None
    wc = wandb.config

    config = Config(
        layers=wc.layers,
        n_heads=wc.n_heads,
        embed_dims=wc.embed_dims,
        dropout=wc.dropout,
        lr=wc.lr,
        warmup_steps=wc.warmup_steps,
        wd=wc.wd,
    )

    # Validate dims divisible by heads
    if config.embed_dims % config.n_heads != 0:
        wandb.finish(exit_code=1)
        return

    tok = ChessTokenizer.build_complete_vocab()
    id_to_move = {
        tid: tok.id_to_token(tid) for tid in range(tok.vocab_size) if tid not in _SPECIAL_IDS
    }

    batch_size: int = wc.batch_size
    dm = ChessDataModule(PROCESSED_DIR, batch_size)
    model = ChessFormer(config, id_to_move, stockfish_path=STOCKFISH_PATH)
    wandb_logger = WandbLogger(experiment=run)

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        logger=wandb_logger,
        gradient_clip_val=1.0,
    )
    trainer.fit(model, datamodule=dm)

    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--processed-dir",
        type=str,
        required=True,
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs per sweep run",
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default=None,
        help="Path to Stockfish binary",
    )
    args = parser.parse_args()

    PROCESSED_DIR = Path(args.processed_dir)
    EPOCHS = args.epochs
    STOCKFISH_PATH = args.stockfish_path

    sweep_id = os.environ.get("WANDB_SWEEP_ID")
    if sweep_id:
        entity = os.environ.get("WANDB_ENTITY", "")
        sweep_path = f"{entity}/chesstf/{sweep_id}" if entity else f"chesstf/{sweep_id}"
        wandb.agent(sweep_path, function=sweep_train)
    else:
        sweep_train()

from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger

from chesstf.data.tokenizer import SPECIAL_TOKENS, ChessTokenizer
from chesstf.model.config import Config
from chesstf.model.transformer import ChessFormer
from chesstf.training.datamodule import ChessDataModule

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
_SPECIAL_IDS: frozenset[int] = frozenset(SPECIAL_TOKENS.values())


def main(
    config: Config,
    processed_dir: Path,
    batch_size: int,
    epochs: int,
    stockfish_path: str | None,
) -> None:
    tok = ChessTokenizer.build_complete_vocab()
    id_to_move = {
        tid: tok.id_to_token(tid)
        for tid in range(tok.vocab_size)
        if tid not in _SPECIAL_IDS
    }

    wandb_logger = WandbLogger(project="chesstf")

    dm = ChessDataModule(processed_dir, batch_size)
    model = ChessFormer(config, id_to_move, stockfish_path=stockfish_path)
    trainer = L.Trainer(max_epochs=epochs, logger=wandb_logger, gradient_clip_val=1.0)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--processed-dir",
        help="Path to processed data directory. Pass a parent directory to train on all month subdirectories, or a single month directory.",
        type=str,
        required=True
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size",
        type=int,
        default=32
    )
    parser.add_argument(
        "--epochs",
        help="Training epochs",
        type=int,
        default=2
    )
    parser.add_argument(
        "--layers",
        help="Number of transformer layers",
        type=int,
        default=8
    )
    parser.add_argument(
        "--heads",
        help="Number of transformer heads per layer",
        type=int,
        default=4
    )
    parser.add_argument(
        "--dims",
        help="Number of embedding dimensions",
        type=int,
        default=256
    )
    parser.add_argument(
        "--stockfish-path",
        help="Path to Stockfish binary (default: /usr/bin/stockfish)",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    config = Config()
    config.layers = args.layers
    config.n_heads = args.heads
    config.embed_dims = args.dims

    processed_dir = Path(args.processed_dir)

    main(config, processed_dir, args.batch_size, args.epochs, args.stockfish_path)

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as L

from chesstf.model.config import Config
from chesstf.model.transformer import ChessFormer
from chesstf.training.datamodule import ChessDataModule


def main(config: Config, processed_dir: Path, batch_size: int, epochs: int) -> None:
    dm = ChessDataModule(processed_dir, batch_size)
    model = ChessFormer(config)
    trainer = L.Trainer(max_epochs=epochs)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--processed-dir",
        help="Path to processed data directory",
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
        default=2
    )
    parser.add_argument(
        "--heads",
        help="Number of transformer heads per layer",
        type=int,
        default=2
    )
    parser.add_argument(
        "--dims",
        help="Number of embedding dimensions",
        type=int,
        default=256
    )

    args = parser.parse_args()

    config = Config()
    config.layers = args.layers
    config.n_heads = args.heads
    config.embed_dims = args.dims

    processed_dir = Path(args.processed_dir)

    main(config, processed_dir, args.batch_size, args.epochs)
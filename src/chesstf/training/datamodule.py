from __future__ import annotations

from typing import TYPE_CHECKING

import pytorch_lightning as L
from torch.utils.data import DataLoader

from chesstf.data.dataset import ChessDataset

if TYPE_CHECKING:
    from pathlib import Path


class ChessDataModule(L.LightningDataModule):
    def __init__(
        self,
        processed_dir: Path,
        batch_size: int = 256,
        max_seq_len: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> None:
        super().__init__()
        self.processed_dir = processed_dir
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.save_hyperparameters()

    def setup(self, stage: str | None) -> None:
        if stage in ("fit", None):
            self.train_ds = ChessDataset.from_split(
                self.processed_dir, 
                "train",
                self.max_seq_len
            )
            self.val_ds = ChessDataset.from_split(
                self.processed_dir, 
                "val",
                self.max_seq_len
            )

    def train_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        return DataLoader(
            self.train_ds,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        return DataLoader(
            self.val_ds,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )
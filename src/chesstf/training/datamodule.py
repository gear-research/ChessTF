from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytorch_lightning as L
from torch.utils.data import ConcatDataset, DataLoader

from chesstf.data.dataset import ChessDataset

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


def _has_nonempty_split(d: Path) -> bool:
    """Check that train.bin and val.bin exist and are non-empty."""
    return all((d / f).stat().st_size > 0 for f in ("train.bin", "val.bin"))


def _discover_dirs(root: Path) -> list[Path]:
    """Return sorted month directories under root that contain non-empty splits.

    Falls back to root itself if it directly contains train.bin (single-month
    path passed explicitly). Skips directories with empty .bin files.
    """
    if (root / "train.bin").exists():
        if not _has_nonempty_split(root):
            raise FileNotFoundError(f"Empty train.bin or val.bin in {root}")
        return [root]

    candidates = sorted(p.parent for p in root.glob("*/train.bin"))
    dirs = []
    for d in candidates:
        if _has_nonempty_split(d):
            dirs.append(d)
        else:
            log.warning("Skipping %s — empty train.bin or val.bin", d.name)

    if not dirs:
        raise FileNotFoundError(f"No non-empty train.bin files found under {root}")
    return dirs


class ChessDataModule(L.LightningDataModule):
    def __init__(
        self,
        processed_dir: Path,
        batch_size: int = 256,
        max_seq_len: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
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
            dirs = _discover_dirs(self.processed_dir)
            log.info("Loading data from %d director%s under %s", len(dirs), "y" if len(dirs) == 1 else "ies", self.processed_dir)

            train_datasets: list[ChessDataset] = []
            val_datasets: list[ChessDataset] = []
            for d in dirs:
                td = ChessDataset.from_split(d, "train", self.max_seq_len)
                vd = ChessDataset.from_split(d, "val", self.max_seq_len)
                log.info("  %s — train: %d games, val: %d games", d.name, len(td), len(vd))
                train_datasets.append(td)
                val_datasets.append(vd)

            self.train_ds: ConcatDataset[dict] = ConcatDataset(train_datasets)
            self.val_ds: ConcatDataset[dict] = ConcatDataset(val_datasets)
            log.info("Total — train: %d games, val: %d games", len(self.train_ds), len(self.val_ds))

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
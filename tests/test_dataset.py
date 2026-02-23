"""Tests for dataset.py — creates .bin/.idx files programmatically."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from chesstf.data.dataset import _IDX_DTYPE, _TOKEN_DTYPE, ChessDataset, encode_to_binary
from chesstf.data.tokenizer import ChessTokenizer

if TYPE_CHECKING:
    from pathlib import Path


def _make_tokenizer(moves: list[str] | None = None) -> ChessTokenizer:
    tok = ChessTokenizer()
    if moves is None:
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "b2b4", "c5b4", "d2d4"]
    tok.build_vocab([moves])
    return tok


def _write_bin_idx(
    tmp_path: Path,
    split: str,
    sequences: list[list[int]],
) -> tuple[Path, Path]:
    """Manually write .bin and .idx files for testing ChessDataset."""
    flat: list[int] = []
    offsets: list[tuple[int, int]] = []
    offset = 0
    for seq in sequences:
        offsets.append((offset, len(seq)))
        flat.extend(seq)
        offset += len(seq)

    bin_path = tmp_path / f"{split}.bin"
    idx_path = tmp_path / f"{split}.idx"

    np.array(flat, dtype=_TOKEN_DTYPE).tofile(bin_path)
    np.array(offsets, dtype=_IDX_DTYPE).tofile(idx_path)
    return bin_path, idx_path


class TestChessDataset:
    def test_len_returns_number_of_games(self, tmp_path: Path) -> None:
        seqs = [[1, 5, 6, 7, 2], [1, 8, 9, 2]]
        bin_p, idx_p = _write_bin_idx(tmp_path, "train", seqs)
        ds = ChessDataset(bin_p, idx_p, max_seq_len=16)
        assert len(ds) == 2

    def test_getitem_returns_tensors(self, tmp_path: Path) -> None:
        seqs = [[1, 5, 6, 7, 2]]
        bin_p, idx_p = _write_bin_idx(tmp_path, "train", seqs)
        ds = ChessDataset(bin_p, idx_p, max_seq_len=16)
        sample = ds[0]
        assert "input_ids" in sample
        assert "labels" in sample
        assert isinstance(sample["input_ids"], torch.Tensor)
        assert isinstance(sample["labels"], torch.Tensor)

    def test_input_ids_shape(self, tmp_path: Path) -> None:
        seqs = [[1, 5, 6, 7, 2]]
        bin_p, idx_p = _write_bin_idx(tmp_path, "train", seqs)
        max_seq = 16
        ds = ChessDataset(bin_p, idx_p, max_seq_len=max_seq)
        sample = ds[0]
        assert sample["input_ids"].shape == (max_seq,)
        assert sample["labels"].shape == (max_seq,)

    def test_input_ids_dtype_is_int64(self, tmp_path: Path) -> None:
        seqs = [[1, 5, 6, 7, 2]]
        bin_p, idx_p = _write_bin_idx(tmp_path, "train", seqs)
        ds = ChessDataset(bin_p, idx_p, max_seq_len=16)
        sample = ds[0]
        assert sample["input_ids"].dtype == torch.int64
        assert sample["labels"].dtype == torch.int64

    def test_padding_with_pad_id(self, tmp_path: Path) -> None:
        seq = [1, 5, 6, 2]  # length 4
        bin_p, idx_p = _write_bin_idx(tmp_path, "train", [seq])
        max_seq = 10
        pad_id = 0
        ds = ChessDataset(bin_p, idx_p, max_seq_len=max_seq, pad_id=pad_id)
        sample = ds[0]
        input_ids = sample["input_ids"].tolist()
        # First 4 positions should be the sequence
        assert input_ids[:4] == seq
        # Remaining should be pad
        assert all(v == pad_id for v in input_ids[4:])

    def test_labels_are_shifted_by_one(self, tmp_path: Path) -> None:
        seq = [1, 5, 6, 7, 2]
        bin_p, idx_p = _write_bin_idx(tmp_path, "train", [seq])
        ds = ChessDataset(bin_p, idx_p, max_seq_len=16)
        sample = ds[0]
        labels = sample["labels"].tolist()
        # Label at position i should be token at position i+1
        assert labels[0] == seq[1]
        assert labels[1] == seq[2]
        assert labels[2] == seq[3]
        assert labels[3] == seq[4]

    def test_labels_padding_is_minus_100(self, tmp_path: Path) -> None:
        seq = [1, 5, 6, 2]  # length 4
        bin_p, idx_p = _write_bin_idx(tmp_path, "train", [seq])
        ds = ChessDataset(bin_p, idx_p, max_seq_len=10)
        sample = ds[0]
        labels = sample["labels"].tolist()
        # After the game ends, labels should be -100
        assert all(v == -100 for v in labels[4:])

    def test_truncation_at_max_seq_len(self, tmp_path: Path) -> None:
        seq = list(range(100))
        bin_p, idx_p = _write_bin_idx(tmp_path, "train", [seq])
        max_seq = 20
        ds = ChessDataset(bin_p, idx_p, max_seq_len=max_seq)
        sample = ds[0]
        assert sample["input_ids"].shape == (max_seq,)
        assert sample["labels"].shape == (max_seq,)

    def test_from_split_convenience(self, tmp_path: Path) -> None:
        seqs = [[1, 5, 6, 2], [1, 7, 8, 2]]
        _write_bin_idx(tmp_path, "train", seqs)
        ds = ChessDataset.from_split(tmp_path, "train", max_seq_len=16)
        assert len(ds) == 2

    def test_multiple_games_independent(self, tmp_path: Path) -> None:
        seqs = [[1, 5, 6, 2], [1, 7, 8, 9, 2]]
        bin_p, idx_p = _write_bin_idx(tmp_path, "train", seqs)
        ds = ChessDataset(bin_p, idx_p, max_seq_len=16)
        s0 = ds[0]["input_ids"].tolist()
        s1 = ds[1]["input_ids"].tolist()
        # First tokens should differ
        assert s0[:4] == [1, 5, 6, 2]
        assert s1[:5] == [1, 7, 8, 9, 2]


class TestEncodeToBinary:
    def _make_jsonl(self, tmp_path: Path, records: list[dict]) -> Path:
        import json

        p = tmp_path / "games.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in records))
        return p

    def test_creates_bin_and_idx_files(self, tmp_path: Path) -> None:
        tok = _make_tokenizer()
        records = [
            {"result": "1-0", "white_elo": 2000, "black_elo": 1900, "moves": ["e2e4", "e7e5"]},
            {"result": "0-1", "white_elo": 2000, "black_elo": 1900, "moves": ["e2e4", "e7e5", "g1f3"]},
        ]
        jsonl = self._make_jsonl(tmp_path, records)
        out = tmp_path / "processed"

        encode_to_binary(jsonl, out, tok, val_fraction=0.5, seed=0)

        assert (out / "train.bin").exists()
        assert (out / "train.idx").exists()
        assert (out / "val.bin").exists()
        assert (out / "val.idx").exists()

    def test_returns_counts(self, tmp_path: Path) -> None:
        tok = _make_tokenizer()
        records = [
            {"result": "1-0", "white_elo": 2000, "black_elo": 1900, "moves": ["e2e4", "e7e5"]},
            {"result": "1-0", "white_elo": 2000, "black_elo": 1900, "moves": ["e2e4", "e7e5"]},
            {"result": "1-0", "white_elo": 2000, "black_elo": 1900, "moves": ["e2e4", "e7e5"]},
            {"result": "1-0", "white_elo": 2000, "black_elo": 1900, "moves": ["e2e4", "e7e5"]},
        ]
        jsonl = self._make_jsonl(tmp_path, records)
        out = tmp_path / "processed"

        counts = encode_to_binary(jsonl, out, tok, val_fraction=0.25, seed=0)

        assert counts["train"] + counts["val"] == 4

    def test_result_conditioning_prepends_token(self, tmp_path: Path) -> None:
        tok = _make_tokenizer()
        # Use enough records so train split is non-empty (val_fraction=0.25 → 1 val, 3 train)
        records = [
            {"result": "1-0", "white_elo": 2000, "black_elo": 1900, "moves": ["e2e4", "e7e5"]},
            {"result": "1-0", "white_elo": 2000, "black_elo": 1900, "moves": ["e2e4", "e7e5"]},
            {"result": "1-0", "white_elo": 2000, "black_elo": 1900, "moves": ["e2e4", "e7e5"]},
            {"result": "1-0", "white_elo": 2000, "black_elo": 1900, "moves": ["e2e4", "e7e5"]},
        ]
        jsonl = self._make_jsonl(tmp_path, records)
        out = tmp_path / "processed"

        encode_to_binary(jsonl, out, tok, val_fraction=0.25, result_conditioning=True, seed=0)

        idx = np.fromfile(out / "train.idx", dtype=_IDX_DTYPE).reshape(-1, 2)
        tokens = np.fromfile(out / "train.bin", dtype=_TOKEN_DTYPE)

        start, length = int(idx[0, 0]), int(idx[0, 1])
        seq = tokens[start : start + length].tolist()
        # With result conditioning: [w_win_id=3, bos_id=1, ..., eos_id=2]
        assert seq[0] == 3  # <w_win>
        assert seq[1] == 1  # <bos>
        assert seq[-1] == 2  # <eos>

    def test_no_result_conditioning(self, tmp_path: Path) -> None:
        tok = _make_tokenizer()
        records = [
            {"result": "1-0", "white_elo": 2000, "black_elo": 1900, "moves": ["e2e4", "e7e5"]},
            {"result": "1-0", "white_elo": 2000, "black_elo": 1900, "moves": ["e2e4", "e7e5"]},
            {"result": "1-0", "white_elo": 2000, "black_elo": 1900, "moves": ["e2e4", "e7e5"]},
            {"result": "1-0", "white_elo": 2000, "black_elo": 1900, "moves": ["e2e4", "e7e5"]},
        ]
        jsonl = self._make_jsonl(tmp_path, records)
        out = tmp_path / "processed"

        encode_to_binary(jsonl, out, tok, val_fraction=0.25, result_conditioning=False, seed=0)

        idx = np.fromfile(out / "train.idx", dtype=_IDX_DTYPE).reshape(-1, 2)
        tokens = np.fromfile(out / "train.bin", dtype=_TOKEN_DTYPE)
        start, length = int(idx[0, 0]), int(idx[0, 1])
        seq = tokens[start : start + length].tolist()
        # Without conditioning: [bos_id=1, ..., eos_id=2]
        assert seq[0] == 1  # <bos>
        assert seq[-1] == 2  # <eos>

    def test_dataset_readable_after_encode(self, tmp_path: Path) -> None:
        tok = _make_tokenizer()
        records = [
            {"result": "1-0", "white_elo": 2000, "black_elo": 1900, "moves": ["e2e4", "e7e5"]},
            {"result": "0-1", "white_elo": 2000, "black_elo": 1900, "moves": ["e2e4", "e7e5", "g1f3"]},
            {"result": "1-0", "white_elo": 2000, "black_elo": 1900, "moves": ["e2e4", "e7e5"]},
            {"result": "1-0", "white_elo": 2000, "black_elo": 1900, "moves": ["e2e4", "e7e5"]},
        ]
        jsonl = self._make_jsonl(tmp_path, records)
        out = tmp_path / "processed"

        counts = encode_to_binary(jsonl, out, tok, val_fraction=0.25, seed=42)

        ds = ChessDataset.from_split(out, "train", max_seq_len=32)
        assert len(ds) == counts["train"]
        sample = ds[0]
        assert sample["input_ids"].shape == (32,)
        assert sample["labels"].shape == (32,)

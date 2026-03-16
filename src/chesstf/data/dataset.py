"""Binary encoding and PyTorch Dataset for chess game sequences."""

from __future__ import annotations

import contextlib
import json
import random
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    import numpy.typing as npt

    from chesstf.data.tokenizer import ChessTokenizer

# dtype constants
_TOKEN_DTYPE = np.int16
_IDX_DTYPE = np.uint64


def encode_to_binary(
    jsonl_path: Path,
    output_dir: Path,
    tokenizer: ChessTokenizer,
    *,
    val_fraction: float = 0.02,
    result_conditioning: bool = True,
    seed: int = 42,
) -> dict[str, int]:
    """Encode a JSONL file of filtered games into flat binary token arrays.

    Produces ``{output_dir}/train.bin``, ``train.idx``, ``val.bin``, ``val.idx``.

    Binary format
    ~~~~~~~~~~~~~
    - ``{split}.bin``  — flat ``int16`` array of token IDs (2 bytes each).
    - ``{split}.idx``  — ``uint64`` array of shape ``(N, 2)`` where each row
      is ``(start_token_offset, length)`` for one game sequence.

    The encoded sequence for each game is::

        [<result_token>,] <bos> move0 move1 ... moveN <eos>

    If *result_conditioning* is True, the result token (``<w_win>`` / ``<b_win>``
    / ``<draw>``) is prepended before ``<bos>``.

    Args:
        jsonl_path: Path to the filtered JSONL file.
        output_dir: Directory in which output files are written.
        tokenizer: Fitted :class:`ChessTokenizer` with all required moves.
        val_fraction: Fraction of games assigned to the validation split.
        result_conditioning: Whether to prepend the result token.
        seed: RNG seed for reproducible train/val split.

    Returns:
        Dict with keys ``"train"`` and ``"val"`` mapping to game counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all records
    records: list[dict[str, object]] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Shuffle for reproducible split
    rng = random.Random(seed)
    rng.shuffle(records)

    n_val = max(1, int(len(records) * val_fraction))
    splits: dict[str, list[dict[str, object]]] = {
        "val": records[:n_val],
        "train": records[n_val:],
    }

    counts: dict[str, int] = {}
    for split_name, split_records in splits.items():
        tokens_list: list[list[int]] = []
        for record in split_records:
            moves: list[str] = record["moves"]  # type: ignore[assignment]
            result: str = record["result"]  # type: ignore[assignment]

            try:
                encoded = tokenizer.encode(moves, add_special=True)
            except KeyError:
                # Skip games with moves not in vocabulary (shouldn't happen
                # post vocab-build, but be defensive)
                continue

            if result_conditioning:
                with contextlib.suppress(KeyError):
                    encoded = tokenizer.encode(moves, add_special=True, result=result)

            tokens_list.append(encoded)

        _write_split(output_dir, split_name, tokens_list)
        counts[split_name] = len(tokens_list)

    return counts


def _write_split(output_dir: Path, split: str, tokens_list: list[list[int]]) -> None:
    """Write .bin and .idx files for one split."""
    # Build index: (start_offset, length) for each game
    offsets: list[tuple[int, int]] = []
    flat: list[int] = []
    offset = 0
    for seq in tokens_list:
        offsets.append((offset, len(seq)))
        flat.extend(seq)
        offset += len(seq)

    bin_path = output_dir / f"{split}.bin"
    idx_path = output_dir / f"{split}.idx"

    arr = np.array(flat, dtype=_TOKEN_DTYPE)
    np.save(bin_path.with_suffix(""), arr)  # saves as .npy; rename below
    # numpy .save appends .npy; work around by writing raw bytes
    arr.tofile(bin_path)

    idx_arr = np.array(offsets, dtype=_IDX_DTYPE)
    idx_arr.tofile(idx_path)


class ChessDataset(Dataset[dict[str, torch.Tensor]]):
    """Memory-mapped PyTorch dataset over encoded chess game sequences.

    Each sample is a dict with:

    - ``input_ids``: ``int64`` tensor of shape ``(max_seq_len,)`` — token IDs
      padded with ``<pad>`` (id=0).
    - ``labels``: ``int64`` tensor of shape ``(max_seq_len,)`` — next-token
      targets; positions beyond the game end (and the first token) are ``-100``
      so that :class:`torch.nn.CrossEntropyLoss` ignores them.

    The label at position ``i`` is the token at position ``i+1`` in the
    sequence (teacher-forcing / causal LM objective).

    Args:
        bin_path: Path to the ``.bin`` file (raw ``int16`` token IDs).
        idx_path: Path to the ``.idx`` file (``uint64`` offset/length pairs).
        max_seq_len: Context window size; sequences are truncated or padded to
            this length.
        pad_id: Token ID used for padding (default 0 = ``<pad>``).
    """

    def __init__(
        self,
        bin_path: Path,
        idx_path: Path,
        max_seq_len: int = 256,
        pad_id: int = 0,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        # Memory-map the token array
        n_tokens = bin_path.stat().st_size // np.dtype(_TOKEN_DTYPE).itemsize
        self._tokens: npt.NDArray[np.integer[Any]] = np.memmap(
            bin_path, dtype=_TOKEN_DTYPE, mode="r", shape=(n_tokens,)
        )

        # Load the full index into RAM (small: 16 bytes × N games)
        raw_idx = np.fromfile(idx_path, dtype=_IDX_DTYPE)
        self._index: npt.NDArray[np.integer[Any]] = raw_idx.reshape(-1, 2)  # (N, 2)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start, length = int(self._index[idx, 0]), int(self._index[idx, 1])

        # Read raw tokens and cast to int64
        raw: npt.NDArray[np.integer[Any]] = np.array(
            self._tokens[start : start + length], dtype=np.int64
        )

        # Truncate to max_seq_len
        raw = raw[: self.max_seq_len + 1]  # +1 so we can form input/label pair
        seq_len = len(raw)

        # input_ids: tokens[0..seq_len-1], padded to max_seq_len
        input_len = min(seq_len, self.max_seq_len)
        input_ids = np.full(self.max_seq_len, self.pad_id, dtype=np.int64)
        input_ids[:input_len] = raw[:input_len]

        # labels: tokens[1..seq_len], padded with -100
        labels = np.full(self.max_seq_len, -100, dtype=np.int64)
        label_src = raw[1 : input_len + 1]
        labels[: len(label_src)] = label_src

        return {
            "input_ids": torch.from_numpy(input_ids),
            "labels": torch.from_numpy(labels),
        }

    @classmethod
    def from_split(
        cls,
        processed_dir: Path,
        split: str,
        max_seq_len: int = 256,
        pad_id: int = 0,
    ) -> ChessDataset:
        """Convenience constructor: load train or val split from a directory."""
        return cls(
            bin_path=processed_dir / f"{split}.bin",
            idx_path=processed_dir / f"{split}.idx",
            max_seq_len=max_seq_len,
            pad_id=pad_id,
        )

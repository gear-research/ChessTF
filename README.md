# Chess Transformer

A decoder-only transformer that plays chess using Standard Algebraic Notation (SAN). End-to-end ML engineering project demonstrating production-grade practices.

## Setup

**Requirements:** Python 3.11+, pip

```bash
pip install -e ".[dev]"
```

## Data Pipeline

Games flow through three stages:

```
data/raw/*.pgn.zst  →  data/interim/*.jsonl  →  data/processed/{train,val}.{bin,idx}
                   filter.py             tokenizer.py + dataset.encode_to_binary()
```

### Data directory layout

```
data/
  raw/          # Downloaded Lichess PGN dumps (.pgn.zst)
  interim/      # Filtered games as JSONL (one game per line)
  processed/
    vocab.json  # SAN vocabulary
    train.bin   # Flat int16 token array (training split)
    train.idx   # Offset/length index (uint64, shape N×2)
    val.bin
    val.idx
```

### Commands

```bash
# Download a monthly dump
make download YEAR=2023 MONTH=1

# Run the full pipeline (download → filter → vocab → encode)
make process YEAR=2023 MONTH=1

# Dry-run filter: print stats without writing any files
make filter-dry YEAR=2023 MONTH=1
```

All steps are also available as subcommands:

```bash
python -m chesstf.data.process --help
python -m chesstf.data.process download --year 2023 --month 1
python -m chesstf.data.process filter  --year 2023 --month 1 --dry-run
python -m chesstf.data.process vocab   --year 2023 --month 1
python -m chesstf.data.process encode  --year 2023 --month 1
python -m chesstf.data.process full    --year 2023 --month 1
```

## Configuration

Edit `configs/data/default.yaml` to change filtering and encoding parameters:

| Parameter | Default | Description |
|---|---|---|
| `min_elo` | 1800 | Minimum ELO for both players |
| `min_base_time_seconds` | 480 | Minimum base time (8 min = rapid+) |
| `min_moves` | 10 | Minimum number of half-moves |
| `val_fraction` | 0.02 | Fraction of games for validation |
| `max_seq_len` | 256 | Maximum token sequence length |
| `result_conditioning` | true | Prepend result token to sequences |

## Development

```bash
make lint    # ruff + mypy
make format  # auto-fix formatting
make test    # pytest
```

## Architecture

See [`SPEC.md`](SPEC.md) and [`CLAUDE.md`](CLAUDE.md) for full architecture details.

### Token vocabulary

Special tokens occupy fixed IDs:

| Token | ID |
|---|---|
| `<pad>` | 0 |
| `<bos>` | 1 |
| `<eos>` | 2 |
| `<w_win>` | 3 |
| `<b_win>` | 4 |
| `<draw>` | 5 |

SAN move tokens start at ID 6, assigned alphabetically.

### Binary format

- **`.bin`**: flat `int16` numpy array, token IDs back-to-back
- **`.idx`**: `uint64` array of shape `(N, 2)` — `(start_offset, length)` per game

`ChessDataset` memory-maps `.bin` and loads the full `.idx` into RAM. Each `__getitem__` returns `input_ids` and `labels` tensors padded/truncated to `max_seq_len`.

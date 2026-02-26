# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chess Transformer: an end-to-end ML engineering project building a decoder-only transformer that plays chess using UCI move notation. The primary goal is demonstrating production-grade ML engineering practices. The finished product is an interactive chess page on a personal website hosted on AWS. Budget ceiling: $100 (target $10–25).

**Current status:** Data pipeline and model architecture complete. Training ready to run locally. Serving, frontend, and experiment tracking not yet implemented.

## Commands

```bash
make install      # pip install -e ".[dev]"
make test         # pytest tests/ -v
make lint         # ruff check + mypy (strict)
make format       # ruff format + ruff --fix
make download     # Download one Lichess month (YEAR=, MONTH=)
make process      # Full pipeline: download+filter+encode (YEAR=, MONTH=)
make filter-dry   # Dry-run filter stats (YEAR=, MONTH=)
```

Direct CLI entry point:
```bash
python -m chesstf.data.process [download|filter|vocab|encode|full] ...
python -m chesstf.training.trainer [--batch_size N] [--epochs N] [--layers N] ...
```

CI/CD (GitHub Actions — `.github/workflows/ci.yml`):
- **All pushes/PRs:** `ruff` (lint), `mypy` (type check), `pytest` (unit tests), smoke test (tokenizer + CLI help)
- **Not yet wired:** Docker build, CDK/Terraform deploy, integration tests

## Architecture

### Model

Decoder-only transformer with RoPE positional encoding. Default config: 8 layers, 4 heads, 256 embed dim, 256-token context. Vocabulary is 1,974 tokens: 6 special (`<bos>`, `<eos>`, `<pad>`, `<w_win>`, `<b_win>`, `<draw>`) with fixed IDs 0–5, then ~1,968 UCI move tokens sorted alphabetically. Embeddings are weight-tied to the output projection.

**Implemented training regimes:**
1. Supervised pretraining on human games (MVP — ready to run)
2. Result-conditioned pretraining (prepend `<w_win>`/`<b_win>`/`<draw>`; configurable via `result_conditioning` in `configs/data/default.yaml`)

**Planned (stretch):**
3. SL pretraining + RL self-play fine-tuning
4. ELO-conditioned pretraining

**Grammar constraint:** No masking during training. At inference, use `python-chess` to enumerate legal moves and mask logits before sampling.

**Architecture details:**
- `TransformerBlock`: RMS norm, multi-head causal attention (`F.scaled_dot_product_attention` with `is_causal=True`), FFN
- `ChessFormer`: `LightningModule` wrapping the transformer; cross-entropy loss with `ignore_index=-100` for padding

### Data Pipeline

- Source: Lichess monthly PGN dumps (zstd-compressed)
- Filter: both players ELO ≥ 1800, base time ≥ 480s (rapid+), Normal termination, ≥10 half-moves
- Tokenization: **UCI move-level** (e.g. `e2e4`, `g1f3`, `e7e8q`) — no board state needed during tokenization
- Storage: memory-mapped binary `.bin` (int16 tokens) + `.idx` (uint64 offset/length pairs)
- Streaming support: `stream.py` downloads + filters directly from HTTP without writing raw files to disk
- 12 months of processed data already cached: `data/processed/2015-01/` through `2016-06/`
- Config: `configs/data/default.yaml`

### Serving Stack (not yet implemented)

```
Browser → CloudFront → API Gateway → AWS Lambda
                                       ├── ONNX Runtime (CPU inference)
                                       ├── python-chess (legal move generation)
                                       └── FastAPI via Mangum adapter
```

**`POST /move`** — accepts `{ "moves": [...], "temperature": 1.0, "top_k": null, "top_p": null }`, returns `{ "move": "e2e4", "probabilities": {...} }`.

Model exported to ONNX for serving. Lambda chosen for its free tier (1M invocations/month) and zero idle cost.

### Infrastructure

Terraform (not CDK) in `infrastructure/terraform/`: S3, Lambda, API Gateway, CloudFront. Not yet wired into CI/CD.

### Repository Structure

```
src/chesstf/
  data/
    download.py     # Lichess monthly dump downloader (streaming + SHA-256)
    filter.py       # FilterConfig, FilterStats, filter_games()
    tokenizer.py    # ChessTokenizer; UCI moves; build_complete_vocab()
    dataset.py      # encode_to_binary(), ChessDataset (memmap .bin + .idx)
    process.py      # CLI: download/filter/vocab/encode/full subcommands
    stream.py       # Stream download+filter directly (no temp disk files)
  model/
    config.py       # Config dataclass (all hyperparameters)
    positional.py   # RotaryPositionalEmbeddings (RoPE)
    transformer.py  # TransformerBlock, ChessFormer (LightningModule)
  training/
    trainer.py      # CLI entry point; argparse; creates DataModule + Trainer
    datamodule.py   # ChessDataModule (LightningDataModule)
configs/
  data/default.yaml # Filter + encoding params
infrastructure/
  terraform/        # main.tf, s3.tf, variables.tf, outputs.tf
tests/
  conftest.py
  test_tokenizer.py
  test_filter.py
  test_download.py
  test_dataset.py
```

**Not yet implemented:** `src/chesstf/serving/` (FastAPI, ONNX inference, Lambda handler), `frontend/`, W&B integration, `src/chesstf/utils/`

## Key Technical Decisions

- **UCI notation throughout** — tokenizer, filter output (JSONL), and dataset all use UCI (not SAN); avoids board state during tokenization
- **PyTorch Lightning** for training (`LightningModule` + `Trainer`; LR scheduling via `configure_optimizers()`)
- **YAML configs** for data/training hyperparameters (no Hydra yet)
- **RMS norm** (not layer norm) in transformer blocks
- **Weight tying** — embedding matrix shared with output projection
- **W&B** free tier for experiment tracking (planned, not yet integrated)
- **ONNX Runtime** (CPU) for Lambda inference — model is tiny, single-digit ms inference
- **python-chess** for board state management and legal move enumeration at inference
- **DVC** for data versioning (configured, S3 remote not yet set up)
- Training containerized with NVIDIA PyTorch base image; sanity-check locally before paid GPU runs (Vast.ai/RunPod)
- Frontend: vanilla HTML/CSS/JS with chessboard.js and chess.js (no framework)

## Lint / Type Rules to Watch

- **TCH rule active** — imports only used in annotations must be in `if TYPE_CHECKING:` block
- **SIM115** — use context managers for `open()`; use `contextlib.ExitStack` for conditional multi-file patterns
- **`from __future__ import annotations`** required in all source files
- Mypy strict mode; `chess.pgn.read_game()` needs `# type: ignore[arg-type]`; pyyaml needs `# type: ignore[import-untyped]`

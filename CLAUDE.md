# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chess Transformer: an end-to-end ML engineering project building a decoder-only transformer that plays chess using Standard Algebraic Notation (SAN). The primary goal is demonstrating production-grade ML engineering practices. The finished product is an interactive chess page on a personal website hosted on AWS. Budget ceiling: $100 (target $10–25).

**Current status:** Early specification phase. Only `SPEC.md` exists. Implementation has not begun.

## Planned Commands

Once implemented, the Makefile will expose:

```bash
make train        # Run training locally
make test         # Run pytest
make lint         # Run ruff and mypy
make deploy       # Deploy infrastructure via CDK
make docker-build # Build training container
```

CI/CD (GitHub Actions):
- **PR:** `ruff` (lint), `mypy` (type check), `pytest` (unit tests), smoke test (forward pass with dummy input)
- **Merge to main:** Docker build/push to ECR, CDK/Terraform deploy, integration test against deployed API

## Architecture

### Model

Decoder-only transformer with RoPE positional encoding. Starting configuration: 2–8 layers, 4–8 attention heads, 128–256 embedding dim, 256-token context, 1–10M parameters. Vocabulary is ~1,800–2,000 SAN move tokens plus special tokens (`<bos>`, `<eos>`, `<pad>`, `<w_win>`, `<b_win>`, `<draw>`, `<elo_N>`).

**Training regimes** (in order of complexity):
1. Supervised pretraining on human games (MVP)
2. Result-conditioned pretraining (prepend `<w_win>`/`<b_win>`/`<draw>` to sequences)
3. SL pretraining + RL self-play fine-tuning (stretch)
4. ELO-conditioned pretraining (stretch)

**Grammar constraint:** No masking during training. At inference, use `python-chess` to enumerate legal moves and mask logits before sampling.

### Data Pipeline

- Source: Lichess monthly PGN dumps (zstd-compressed)
- Filter: both players ELO ≥ 1800, standard/rapid time controls, completed games only
- Tokenization: move-level (each SAN string = 1 token)
- Storage: memory-mapped NumPy `.npy` or Arrow/Parquet
- Versioning: DVC with S3 remote

### Serving Stack

```
Browser → CloudFront → API Gateway → AWS Lambda
                                       ├── ONNX Runtime (CPU inference)
                                       ├── python-chess (legal move generation)
                                       └── FastAPI via Mangum adapter
```

**`POST /move`** — accepts `{ "moves": [...], "temperature": 1.0, "top_k": null, "top_p": null }`, returns `{ "move": "Bc4", "probabilities": {...} }`.

Model exported to ONNX for serving. Lambda chosen for its free tier (1M invocations/month) and zero idle cost.

### Infrastructure

All AWS resources in AWS CDK (Python) or Terraform: S3 (data, artifacts, static site), Lambda + layers, API Gateway, CloudFront, IAM, CloudWatch dashboards/alarms, optional DynamoDB for inference logging.

### Repository Structure (planned)

```
src/
  data/       # download.py, filter.py, tokenizer.py, dataset.py
  model/      # transformer.py, positional.py (RoPE), config.py
  training/   # trainer.py, scheduler.py, evaluate.py
  serving/    # app.py (FastAPI), inference.py (ONNX + masking), handler.py (Mangum)
  utils/      # chess_utils.py, logging.py
configs/      # Hydra/YAML configs for model, training, data, serving
infrastructure/cdk/  # CDK stacks: lambda_stack.py, api_stack.py, monitoring_stack.py
frontend/     # Vanilla HTML/CSS/JS + chessboard.js + chess.js
tests/        # test_tokenizer.py, test_model.py, test_inference.py, test_api.py
```

## Key Technical Decisions

- **PyTorch** for training (raw or PyTorch Lightning)
- **Hydra** or YAML configs for all hyperparameters
- **W&B** free tier for experiment tracking; every run links to its dataset version (DVC hash) and container image digest
- **ONNX Runtime** (CPU) for Lambda inference — model is tiny, single-digit ms inference
- **python-chess** for board state management and legal move enumeration
- **DVC** for data versioning; every dataset version reproducible from raw source + config
- Training containerized with NVIDIA PyTorch base image; sanity-check locally before paid GPU runs (Vast.ai/RunPod)
- Frontend: vanilla HTML/CSS/JS with chessboard.js and chess.js (no framework)

# Chess Transformer: Project Specification

## Overview

This project is a portfolio-oriented, end-to-end machine learning engineering effort. The goal is to design, train, and deploy a custom decoder-only transformer model that plays chess, using algebraic chess notation as its native representation. While the ML task itself is interesting, the primary objective is to demonstrate production-grade ML engineering practices across the full lifecycle: data curation, training infrastructure, experiment tracking, deployment, serving, monitoring, and CI/CD.

The finished product is a page on the author's personal website (static HTML hosted on AWS) where a user can play against the model or watch the model play against itself.

**Budget ceiling:** $100 absolute maximum, target $10–25.

---

## Architecture Decision Record

### Why a Decoder-Only Transformer

1. **Limited token space:** Chess notation has a small, fully enumerable vocabulary (~1,800–2,000 possible SAN move tokens). No need for BPE or a full LLM-scale tokenizer.
2. **Clean signal:** Algebraic notation unambiguously encodes the full game state with minimal redundancy.
3. **Autoregressive nature:** Chess games are inherently sequential — each move depends only on prior moves. Generation is simply alternating half-moves.
4. **Causal masking:** Future moves must not influence the current prediction, making decoder-only (with causal attention) the natural fit.
5. **Small decision space:** Chess is far less complex than natural language, allowing a model in the 1–10M parameter range to be viable. This keeps training and serving costs minimal.
6. **Data availability:** Lichess publishes free, massive monthly PGN database dumps with rich metadata (ELO, time control, result, etc.), making data acquisition trivial.
7. **Grammar-constrained decoding:** Legal moves for any board state are fully determinable. A CFG or logit mask can restrict the model to legal outputs at inference time.

### Model Dimensions (Starting Point)

| Hyperparameter     | Range / Default |
|--------------------|-----------------|
| Layers             | 2–8             |
| Attention heads    | 4–8             |
| Embedding dim      | 128–256         |
| Context length     | 256 tokens      |
| Parameters (est.)  | 1–10M           |
| Positional encoding| RoPE            |

Start with the smallest viable configuration. Confirm learning, then scale up.

---

## Data Pipeline

### Source

- **Lichess open database** (https://database.lichess.org/)
- Format: PGN (Portable Game Notation), compressed with zstd
- Use one or a few monthly dumps initially; scale up as needed

### Filtering Criteria

- Both players ELO ≥ 1800
- Time control: standard or rapid (no bullet/ultrabullet)
- Game completed normally (no abandonment/timeout on move 1)
- Decisive or drawn games only

### Tokenization

**Move-level tokenization** (preferred over character-level):

- Each legal SAN string (e.g., `e4`, `Nf3`, `Bxe5+`, `O-O-O`, `e8=Q#`) is a single token
- Vocabulary size: ~1,800–2,000 tokens
- Typical game length: ~80 tokens (40 full moves)

**Special tokens:**

| Token       | Purpose                                                  |
|-------------|----------------------------------------------------------|
| `<bos>`     | Beginning of game                                        |
| `<eos>`     | End of game                                              |
| `<pad>`     | Padding for batching                                     |
| `<w_win>`   | White wins (prepended to sequence, optional)             |
| `<b_win>`   | Black wins (prepended to sequence, optional)             |
| `<draw>`    | Draw (prepended to sequence, optional)                   |
| `<elo_N>`   | Optional ELO bucket tokens for conditioning              |

The result-conditioning tokens are optional and represent a training regime choice (see Training Regimes below).

### Processing Pipeline

1. Download raw PGN dump from Lichess
2. Filter games by metadata criteria
3. Parse PGN into move sequences
4. Tokenize and encode as integer sequences
5. Store as memory-mapped files (NumPy `.npy` or Arrow/Parquet) for efficient streaming
6. Track with DVC, remote on S3

### Data Versioning

- **DVC** (Data Version Control) on top of Git
- DVC remote: S3 bucket
- All processing parameters (ELO threshold, time controls, etc.) tracked in DVC pipeline config
- Every dataset version is reproducible from raw source + config

---

## Training

### Framework & Configuration

- **PyTorch** (raw or with PyTorch Lightning)
- **Hydra** or YAML-based config system for all hyperparameters
- Modular code structure: separate data, model, training, and evaluation modules
- All training code containerized via **Dockerfile** (base: NVIDIA PyTorch image)

### Training Regimes (To Explore)

These are not mutually exclusive. They represent a progression of sophistication:

1. **Supervised pretraining only:** Train on human games with standard cross-entropy next-token prediction. Simplest baseline.
2. **Result-conditioned pretraining:** Prepend a result token (`<w_win>`, `<b_win>`, `<draw>`) to each game sequence during training. At inference, always condition on the model winning. This biases the model toward winning lines without any RL.
3. **Supervised pretraining + RL self-play fine-tuning:** Pretrain on human games, then fine-tune with self-play using policy gradient methods. More complex but potentially stronger play.
4. **ELO-conditioned pretraining:** Include ELO bucket tokens so the model learns to play at different skill levels. At inference, condition on a high ELO.

**MVP target: Regime 1 or 2.** Regimes 3 and 4 are stretch goals.

### Compute Strategy

| Stage                  | Platform                        | Estimated Cost |
|------------------------|---------------------------------|----------------|
| Development & debugging| Local machine / Colab free tier | $0             |
| Training runs          | Vast.ai or RunPod (A100/4090)  | $5–15          |
| Evaluation             | Same GPU instance or local      | included above |

- Expect 2–10 hours of total GPU time across all experiments
- Always run a tiny sanity-check training locally before launching paid GPU jobs

### Experiment Tracking

- **Weights & Biases (W&B)** free tier
- Log per run: loss curves, learning rate schedule, gradient norms, evaluation metrics, hyperparameter config, model checkpoints
- Every model artifact linked to its training run and dataset version

### Evaluation Metrics

| Metric                          | Description                                                    |
|---------------------------------|----------------------------------------------------------------|
| Cross-entropy loss              | Standard training loss                                         |
| Move prediction accuracy        | Top-1 and top-5 accuracy on held-out games                     |
| Legal move rate (unmasked)      | % of predictions that are legal moves (before grammar masking) |
| Stockfish move agreement        | % overlap with Stockfish top-N moves at various depth limits   |
| Win/draw/loss vs Stockfish      | Game outcomes against Stockfish at limited search depth         |
| Self-play game quality          | Manual inspection + average game length + result distribution  |

### Grammar Constraint Design

- **Training:** Do NOT mask illegal moves. Let the model learn legality from data. Compute loss over all tokens including illegal predictions.
- **Inference:** Use `python-chess` to generate the legal move set for the current board state. Mask logits to zero out illegal moves before sampling/argmax. This guarantees legal play.
- **Future work:** Explore architectural modifications that provide legal-move hints as input features (e.g., a binary mask or attention bias).

---

## Model Registry & Artifact Management

- Trained model checkpoints stored in **S3** and tracked with **DVC** or **W&B Artifacts**
- Every deployed model must be traceable to:
  - Exact training run ID (W&B)
  - Dataset version (DVC commit hash)
  - Hyperparameter config
  - Training container image digest
- Export final model to **ONNX** format for serving

---

## Serving & Deployment

### Inference Stack (MVP)

```
User Browser
    │
    ▼
CloudFront (CDN, static site)
    │
    ▼
API Gateway (REST)
    │
    ▼
AWS Lambda (Python runtime)
    ├── ONNX Runtime (CPU inference)
    ├── python-chess (legal move generation, board state tracking)
    └── FastAPI via Mangum adapter
```

### API Design

**`POST /move`**
- Request: `{ "moves": ["e4", "e5", "Nf3", ...], "temperature": 1.0, "top_k": null, "top_p": null }`
- Response: `{ "move": "Bc4", "probabilities": {"Bc4": 0.35, "Bb5": 0.28, ...} }`
- The server reconstructs the board state from the move list, generates legal moves, runs inference, masks illegal moves, samples, and returns the result.

**`POST /self-play`** (stretch goal)
- Request: `{ "num_moves": 80, "temperature": 1.0 }`
- Response: `{ "moves": ["e4", "e5", ...], "result": "1-0" }`

### Why Lambda

- Model is tiny (few MB ONNX); inference is single-digit ms on CPU
- Lambda free tier: 1M invocations/month
- Pay-per-use: zero cost when nobody is playing
- Cold starts acceptable for this use case (~1–2s worst case)

### Deployment Alternatives (For Later Portfolio Enhancement)

- **ECS Fargate:** Persistent container, more control, higher baseline cost
- **SageMaker Serverless Inference:** More "enterprise" pattern, higher cost, overkill for this model

---

## Infrastructure as Code

All AWS resources defined in **AWS CDK (Python)** or **Terraform**:

- S3 buckets (data, model artifacts, website hosting)
- Lambda function + layers
- API Gateway (REST API)
- CloudFront distribution
- IAM roles and policies
- CloudWatch dashboards and alarms
- DynamoDB table (if used for inference logging)

Everything version-controlled in Git. Infrastructure is reproducible from code.

---

## CI/CD

**Platform:** GitHub Actions

### On Pull Request

- Lint (ruff)
- Type check (mypy)
- Unit tests (pytest)
- Smoke test: forward pass through the model with dummy input

### On Merge to Main

- Build and push training Docker image (to ECR or Docker Hub)
- Deploy updated Lambda function and infrastructure (via CDK/Terraform)
- Run integration test against deployed API

### Optional / Stretch

- Scheduled monthly pipeline: download latest Lichess dump, retrain, evaluate, deploy if improved
- Model quality gate: block deployment if evaluation metrics regress

---

## Monitoring & Observability

### Infrastructure Monitoring (CloudWatch)

- Lambda: invocation count, duration, error rate, cold start frequency, throttles
- API Gateway: request count, latency (p50/p95/p99), 4xx/5xx rates
- CloudWatch dashboard aggregating all key metrics

### Model Monitoring

Log per inference request (to CloudWatch Logs or DynamoDB):

- Input sequence length
- Output move
- Probability distribution entropy (high entropy = uncertain model)
- Top-3 move probabilities
- Whether the move required grammar masking (i.e., would the model's top unmasked choice have been illegal?)
- Latency breakdown (model inference vs. board state construction vs. total)

### Alerting

- CloudWatch Alarms on: error rate > threshold, p99 latency > threshold, Lambda concurrent executions near limit
- Notifications via SNS → email

---

## Frontend

### Tech Stack

- Vanilla HTML/CSS/JS (consistent with existing static site)
- **chessboard.js** (or cm-chessboard) for board rendering and piece interaction
- **chess.js** for client-side game logic, move validation, PGN handling
- Hosted on S3 via CloudFront (existing setup)

### Features (MVP)

- Interactive chessboard: user plays white (or choice of color)
- API call on each user move to get the model's response
- Move history display in algebraic notation
- Game result detection (checkmate, stalemate, draw)

### Features (Stretch)

- Self-play viewer: watch the model play both sides with configurable speed
- Inference controls: temperature, top-k, top-p sliders
- Probability heatmap: overlay showing the model's probability distribution across legal moves
- Multiple model versions: let the user pick which checkpoint to play against

---

## Repository Structure

```
chess-transformer/
├── README.md
├── pyproject.toml              # Project config, dependencies
├── Dockerfile                  # Training container
├── Makefile                    # Common commands (train, test, deploy, etc.)
│
├── configs/                    # Hydra / YAML configs
│   ├── model/
│   ├── training/
│   ├── data/
│   └── serving/
│
├── src/
│   ├── data/
│   │   ├── download.py         # Lichess PGN download
│   │   ├── filter.py           # Game filtering logic
│   │   ├── tokenizer.py        # Custom move-level tokenizer
│   │   └── dataset.py          # PyTorch Dataset / DataLoader
│   │
│   ├── model/
│   │   ├── transformer.py      # Model architecture
│   │   ├── positional.py       # RoPE implementation
│   │   └── config.py           # Model hyperparameter dataclass
│   │
│   ├── training/
│   │   ├── trainer.py          # Training loop
│   │   ├── scheduler.py        # LR schedule
│   │   └── evaluate.py         # Evaluation harness (Stockfish, metrics)
│   │
│   ├── serving/
│   │   ├── app.py              # FastAPI application
│   │   ├── inference.py        # ONNX inference + grammar masking
│   │   └── handler.py          # Lambda handler (Mangum)
│   │
│   └── utils/
│       ├── chess_utils.py      # python-chess helpers
│       └── logging.py          # Structured logging setup
│
├── infrastructure/
│   ├── cdk/                    # AWS CDK stacks (or Terraform)
│   │   ├── app.py
│   │   ├── lambda_stack.py
│   │   ├── api_stack.py
│   │   └── monitoring_stack.py
│   └── docker/
│       └── training.Dockerfile
│
├── frontend/
│   ├── index.html
│   ├── chess.js                # Game logic + API integration
│   └── styles.css
│
├── tests/
│   ├── test_tokenizer.py
│   ├── test_model.py
│   ├── test_inference.py
│   └── test_api.py
│
├── notebooks/                  # Exploration / analysis
│   └── data_exploration.ipynb
│
├── dvc.yaml                    # DVC pipeline definition
├── dvc.lock
├── .github/
│   └── workflows/
│       ├── ci.yml              # PR checks
│       └── deploy.yml          # Deployment pipeline
│
└── .dvc/
    └── config                  # DVC remote config (S3)
```

---

## Phase Plan

### Phase 1: Data Pipeline
- Download and parse Lichess PGN data
- Implement filtering and tokenization
- Create PyTorch Dataset with streaming from memory-mapped files
- Set up DVC tracking with S3 remote
- **Deliverable:** Reproducible dataset pipeline, versioned data in S3

### Phase 2: Model & Training (Local)
- Implement transformer architecture with RoPE
- Write training loop with config management
- Train smallest viable model locally, confirm learning (loss decreasing, legal move accuracy improving)
- Set up W&B logging
- **Deliverable:** Working training pipeline, initial experiment results in W&B

### Phase 3: Cloud Training
- Containerize training code
- Run training on Vast.ai / RunPod
- Hyperparameter search (model size, learning rate, etc.)
- Evaluate against Stockfish
- Export best model to ONNX
- **Deliverable:** Trained model checkpoint with full lineage, ONNX artifact

### Phase 4: Serving Infrastructure
- Build FastAPI inference API with grammar-constrained decoding
- Package as Lambda function
- Define infrastructure with CDK/Terraform
- Deploy and test
- **Deliverable:** Live API endpoint returning legal chess moves

### Phase 5: Frontend
- Build chess UI with chessboard.js
- Integrate with API
- Deploy to existing static site
- **Deliverable:** Playable chess page on personal website

### Phase 6: CI/CD & Monitoring
- GitHub Actions pipelines for test, build, deploy
- CloudWatch dashboards and alarms
- Inference logging
- **Deliverable:** Automated deployment, observable production system

### Phase 7: Stretch Goals (Optional)
- Result-conditioned or ELO-conditioned training
- Self-play RL fine-tuning
- Self-play viewer on frontend
- Probability heatmap visualization
- Temperature / sampling parameter controls in UI
- Scheduled retraining pipeline

---

## Key Design Principles

1. **Reproducibility:** Every training run, dataset, and deployment is traceable and reproducible from version-controlled code and config.
2. **Containerization:** Training runs in containers. No "works on my machine."
3. **Infrastructure as code:** No manual AWS console clicking. Everything is CDK/Terraform.
4. **Separation of concerns:** Data, model, training, serving, and infrastructure are independent modules.
5. **Budget discipline:** Use free tiers aggressively. Pay only for GPU time and minimal S3/Lambda usage.
6. **Iterative development:** Start with the smallest viable version of each component. Validate, then scale.
7. **Production patterns at portfolio scale:** The project won't handle production traffic, but the patterns (monitoring, CI/CD, artifact lineage, IaC) should be the same ones a real team would use.
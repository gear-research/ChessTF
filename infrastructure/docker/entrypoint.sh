#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/app/data/processed}"

if [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo "No data found in $DATA_DIR — running dvc pull..."
    dvc pull
else
    echo "Data found in $DATA_DIR — skipping dvc pull."
fi

STOCKFISH_ARG=()
if STOCKFISH_BIN=$(command -v stockfish 2>/dev/null); then
    echo "Found stockfish at $STOCKFISH_BIN"
    STOCKFISH_ARG=(--stockfish-path "$STOCKFISH_BIN")
fi

if [ "${SWEEP_MODE:-}" = "1" ]; then
    echo "Starting W&B sweep agent..."
    exec python scripts/sweep_train.py "${STOCKFISH_ARG[@]}" "$@"
else
    exec python -m chesstf.training.trainer "${STOCKFISH_ARG[@]}" "$@"
fi

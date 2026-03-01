#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/app/data/processed}"

if [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo "No data found in $DATA_DIR — running dvc pull..."
    dvc pull
else
    echo "Data found in $DATA_DIR — skipping dvc pull."
fi

exec python -m chesstf.training.trainer "$@"

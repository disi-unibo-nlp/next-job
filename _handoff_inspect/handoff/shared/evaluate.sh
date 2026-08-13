#!/bin/bash
# Score a checkpoint on the shared evaluation set.
#
# Wraps evaluate.py. MODEL is required; BACKEND picks the encoder path
# (`dense` for recipes 01/03/04, `late-interaction` for recipe 02).
#
# Run it once on the base model before training to get the baseline row, then
# again on the trained checkpoint.
#
# Usage:
#   MODEL=sentence-transformers/all-MiniLM-L6-v2 bash evaluate.sh
#   MODEL=../checkpoints/sentence_transformer/final bash evaluate.sh
#   MODEL=../checkpoints/late_interaction/final BACKEND=late-interaction bash evaluate.sh
#   MODEL=../checkpoints/qwen3_embedding/merged bash evaluate.sh --pooling lasttoken --normalize

set -euo pipefail

cd "$(dirname "$0")"

MODEL="${MODEL:-}"
BACKEND="${BACKEND:-dense}"
DATA_DIR="${DATA_DIR:-../data}"
BATCH_SIZE="${BATCH_SIZE:-32}"
RESULTS_DIR="${RESULTS_DIR:-../results}"

if [ -z "$MODEL" ]; then
    echo "ERROR: set MODEL to a checkpoint directory or Hub model id." >&2
    echo "  MODEL=sentence-transformers/all-MiniLM-L6-v2 bash evaluate.sh" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"
SLUG="$(basename "$MODEL" | tr -c 'A-Za-z0-9._-' '_')"

echo "[$(date +%H:%M:%S)] Evaluating $MODEL (backend=$BACKEND)"

uv run python evaluate.py \
    --backend     "$BACKEND" \
    --model-path  "$MODEL" \
    --data-dir    "$DATA_DIR" \
    --batch-size  "$BATCH_SIZE" \
    --output      "$RESULTS_DIR/${SLUG}.json" \
    "$@"

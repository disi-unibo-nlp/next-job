#!/bin/bash
# Train a dense bi-encoder with Sentence Transformers.
#
# Prerequisite:
#   cd ../shared && bash prepare_data.sh
#
# Memory profile (all-MiniLM-L6-v2, max_seq_length 384, 4 hard negatives):
#   bs=64   ->  ~6 GB     <- default, fits anywhere
#   bs=256  ->  ~22 GB
#   For larger batches without more VRAM use LOSS=cmnrl, which caches gradients
#   and keeps memory flat in the batch size (MINI_BATCH_SIZE is the real knob).
#
# Outputs:
#   ../checkpoints/sentence_transformer/final          the model to evaluate
#   ../checkpoints/sentence_transformer/checkpoints/   trainer checkpoints
#
# Usage:
#   bash run.sh
#   bash run.sh --dry-run
#   LOSS=cmnrl BATCH_SIZE=256 bash run.sh
#   MODEL=BAAI/bge-base-en-v1.5 EPOCHS=2 bash run.sh
#   bash run.sh --max-steps 5 --no-save --no-eval        # smoke run

set -euo pipefail

cd "$(dirname "$0")"

MODEL="${MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
LOSS="${LOSS:-mnrl}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-2e-5}"
NUM_NEGATIVES="${NUM_NEGATIVES:-4}"
DATA_DIR="${DATA_DIR:-../data}"
OUTPUT_DIR="${OUTPUT_DIR:-../checkpoints/sentence_transformer}"
REPORT_TO="${REPORT_TO:-none}"

echo "[$(date +%H:%M:%S)] Training $MODEL with loss=$LOSS"

uv run python train.py \
    --model          "$MODEL" \
    --loss           "$LOSS" \
    --batch-size     "$BATCH_SIZE" \
    --grad-accum     "$GRAD_ACCUM" \
    --epochs         "$EPOCHS" \
    --lr             "$LR" \
    --num-negatives  "$NUM_NEGATIVES" \
    --data-dir       "$DATA_DIR" \
    --output-dir     "$OUTPUT_DIR" \
    --report-to      "$REPORT_TO" \
    "$@"

echo "[$(date +%H:%M:%S)] Done. Evaluate with:"
echo "  cd ../shared && MODEL=$OUTPUT_DIR/final bash evaluate.sh"

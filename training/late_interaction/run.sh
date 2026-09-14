#!/bin/bash
# Train a late-interaction (ColBERT-style) retriever with PyLate.
#
# Prerequisite:
#   cd ../shared && bash prepare_data.sh
#
# Memory profile (GTE-ModernColBERT-v1, document_length 300, 4 negatives,
# gradient checkpointing ON):
#   bs=32   ->  ~14 GB    <- default
#   bs=64   ->  ~26 GB
#   bs=128  ->  ~50 GB
# Gradient checkpointing is effectively required: a multi-vector model keeps
# one embedding per token per document, and each step encodes
# batch_size x (1 + NUM_NEGATIVES) documents.
#
# Keep GRAD_ACCUM at 8 or above. If a batch does not fit, lower BATCH_SIZE or
# NUM_NEGATIVES -- not GRAD_ACCUM. The effective batch (BATCH_SIZE x GRAD_ACCUM)
# is what these settings are tuned around.
#
# Outputs:
#   ../checkpoints/late_interaction/final          the model to evaluate
#   ../checkpoints/late_interaction/checkpoints/   trainer checkpoints
#
# Usage:
#   bash run.sh
#   bash run.sh --dry-run
#   BATCH_SIZE=64 bash run.sh
#   LOSS=cached_contrastive BATCH_SIZE=256 bash run.sh --mini-batch-size 16
#   bash run.sh --max-steps 5 --no-save --no-eval        # smoke run

set -euo pipefail

cd "$(dirname "$0")"

MODEL="${MODEL:-lightonai/GTE-ModernColBERT-v1}"
LOSS="${LOSS:-contrastive}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-3e-6}"
TEMPERATURE="${TEMPERATURE:-0.05}"
NUM_NEGATIVES="${NUM_NEGATIVES:-4}"
DOCUMENT_LENGTH="${DOCUMENT_LENGTH:-300}"
QUERY_LENGTH="${QUERY_LENGTH:-32}"
DATA_DIR="${DATA_DIR:-../data}"
OUTPUT_DIR="${OUTPUT_DIR:-../checkpoints/late_interaction}"
REPORT_TO="${REPORT_TO:-none}"

echo "[$(date +%H:%M:%S)] Training $MODEL with loss=$LOSS"

uv run python train.py \
    --model            "$MODEL" \
    --loss             "$LOSS" \
    --batch-size       "$BATCH_SIZE" \
    --grad-accum       "$GRAD_ACCUM" \
    --epochs           "$EPOCHS" \
    --lr               "$LR" \
    --temperature      "$TEMPERATURE" \
    --num-negatives    "$NUM_NEGATIVES" \
    --document-length  "$DOCUMENT_LENGTH" \
    --query-length     "$QUERY_LENGTH" \
    --data-dir         "$DATA_DIR" \
    --output-dir       "$OUTPUT_DIR" \
    --report-to        "$REPORT_TO" \
    --gradient-checkpointing \
    "$@"

echo "[$(date +%H:%M:%S)] Done. Evaluate with:"
echo "  cd ../shared && MODEL=$OUTPUT_DIR/final BACKEND=late-interaction bash evaluate.sh"

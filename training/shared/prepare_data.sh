#!/bin/bash
# Build the shared dataset every recipe in ../ trains on.
#
# Downloads sentence-transformers/natural-questions, holds out an evaluation
# slice, mines hard negatives, and writes train.jsonl / eval_corpus.jsonl /
# eval_queries.jsonl into ../data/.
#
# Runtime: ~10 min on a GPU for the default 50k rows (mostly the mining
# forward pass over ~50k passages). CPU-only works but is slow; shrink with
# TRAIN_SIZE.
#
# Outputs:
#   ../data/train.jsonl          query / positive / negatives
#   ../data/eval_corpus.jsonl    docid / text
#   ../data/eval_queries.jsonl   qid / query / positive_docids
#   ../data/summary.json         what was built and with which settings
#
# Usage:
#   bash prepare_data.sh
#   TRAIN_SIZE=500 EVAL_QUERIES=100 bash prepare_data.sh      # quick smoke build
#   NUM_NEGATIVES=16 bash prepare_data.sh
#   bash prepare_data.sh --with-scores --overwrite

set -euo pipefail

cd "$(dirname "$0")"

TRAIN_SIZE="${TRAIN_SIZE:-50000}"
EVAL_QUERIES="${EVAL_QUERIES:-2000}"
EVAL_CORPUS_SIZE="${EVAL_CORPUS_SIZE:-20000}"
NUM_NEGATIVES="${NUM_NEGATIVES:-8}"
MINING_MODEL="${MINING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
OUTPUT_DIR="${OUTPUT_DIR:-../data}"

echo "[$(date +%H:%M:%S)] Preparing shared dataset -> $OUTPUT_DIR"

uv run python prepare_data.py \
    --train-size        "$TRAIN_SIZE" \
    --eval-queries      "$EVAL_QUERIES" \
    --eval-corpus-size  "$EVAL_CORPUS_SIZE" \
    --num-negatives     "$NUM_NEGATIVES" \
    --mining-model      "$MINING_MODEL" \
    --output-dir        "$OUTPUT_DIR" \
    "$@"

echo "[$(date +%H:%M:%S)] Done."

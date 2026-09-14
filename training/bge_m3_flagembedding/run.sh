#!/bin/bash
# Train BGE-M3 with FlagEmbedding.
#
# Prerequisite:
#   cd ../shared && bash prepare_data.sh
#
# BGE-M3 is XLM-RoBERTa-large (568M) and --unified-finetuning trains three
# retrieval heads at once, so this is by far the heaviest of the four recipes.
#
# Memory profile (unified finetuning, passage_max_len 512, train_group_size 8,
# DeepSpeed ZeRO stage 0):
#   bs=1  -> ~28 GB
#   bs=2  -> ~46 GB   <- default
#   bs=4  -> ~80 GB
# Add --gradient-checkpointing to roughly halve those at ~30% slowdown.
# The real driver is BATCH_SIZE x (NEG_NUM + 1) passages per step.
#
# Multi-GPU: this script runs single-process. For several GPUs use torchrun,
# which gives negatives_cross_device something to actually share:
#   torchrun --nproc_per_node 4 train.py --batch-size 2 --grad-accum 2
#
# Outputs:
#   ../checkpoints/bge_m3/                 the trained model
#   ../checkpoints/bge_m3/ft_data/         the converted FlagEmbedding JSONL
#
# Usage:
#   bash run.sh
#   bash run.sh --dry-run
#   NEG_NUM=15 BATCH_SIZE=1 bash run.sh --gradient-checkpointing
#   bash run.sh --dense-only              # skip the sparse and colbert heads
#   bash run.sh --max-steps 5             # smoke run

set -euo pipefail

cd "$(dirname "$0")"

MODEL="${MODEL:-BAAI/bge-m3}"
NEG_NUM="${NEG_NUM:-7}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-5}"
PASSAGE_MAX_LEN="${PASSAGE_MAX_LEN:-512}"
QUERY_MAX_LEN="${QUERY_MAX_LEN:-128}"
SAVE_STEPS="${SAVE_STEPS:-50000}"
DEEPSPEED="${DEEPSPEED:-ds_stage0.json}"
DATA_DIR="${DATA_DIR:-../data}"
OUTPUT_DIR="${OUTPUT_DIR:-../checkpoints/bge_m3}"
REPORT_TO="${REPORT_TO:-none}"

echo "[$(date +%H:%M:%S)] Training $MODEL (neg_num=$NEG_NUM, group=$((NEG_NUM + 1)))"

uv run python train.py \
    --model            "$MODEL" \
    --neg-num          "$NEG_NUM" \
    --batch-size       "$BATCH_SIZE" \
    --grad-accum       "$GRAD_ACCUM" \
    --epochs           "$EPOCHS" \
    --lr               "$LR" \
    --passage-max-len  "$PASSAGE_MAX_LEN" \
    --query-max-len    "$QUERY_MAX_LEN" \
    --save-steps       "$SAVE_STEPS" \
    --deepspeed        "$DEEPSPEED" \
    --data-dir         "$DATA_DIR" \
    --output-dir       "$OUTPUT_DIR" \
    --report-to        "$REPORT_TO" \
    "$@"

echo "[$(date +%H:%M:%S)] Done. Evaluate with:"
echo "  cd ../shared && MODEL=$OUTPUT_DIR bash evaluate.sh"

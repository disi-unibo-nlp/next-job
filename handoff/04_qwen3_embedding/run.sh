#!/bin/bash
# Train Qwen3-Embedding with ms-swift, following the upstream QwenLM recipe.
#
# Prerequisite:
#   cd ../shared && bash prepare_data.sh
#
# The backbone is an LLM, so this is the heaviest recipe per parameter. LoRA is
# the default because full fine-tuning an 8B embedder is not practical on a
# single RTX 3090.
#
# Memory profile (Qwen3-Embedding-8B, LoRA r=8, 4-bit, max_length 512,
# 7 negatives, gradient checkpointing ON) is tight on a 24 GB RTX 3090:
#   PER_DEVICE_BATCH_SIZE=4 can OOM when the card is not fully free.
#   PER_DEVICE_BATCH_SIZE=2 is the safer default.
# Each device processes PER_DEVICE_BATCH_SIZE x (1 + NUM_NEGATIVES) texts per
# step, so the negative count moves memory as much as the batch size does.
#
# Full fine-tuning, upstream style (8 GPUs):
#   TRAIN_TYPE=full DEEPSPEED=zero3 NPROC_PER_NODE=8 PER_DEVICE_BATCH_SIZE=4 bash run.sh
#
# Outputs:
#   ../checkpoints/qwen3_embedding/             swift run directories
#   ../checkpoints/qwen3_embedding/swift_data/  the converted swift JSONL
#
# Usage:
#   bash run.sh
#   bash run.sh --dry-run
#   MODEL=Qwen/Qwen3-Embedding-8B PER_DEVICE_BATCH_SIZE=2 bash run.sh
#   bash run.sh --max-steps 5                   # smoke run

set -euo pipefail

cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen3-Embedding-8B}"
TRAIN_TYPE="${TRAIN_TYPE:-lora}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-6e-6}"
NUM_NEGATIVES="${NUM_NEGATIVES:-7}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SAVE_STEPS="${SAVE_STEPS:-50000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
QUANT_METHOD="${QUANT_METHOD:-bnb}"
QUANT_BITS="${QUANT_BITS:-4}"
BNB_4BIT_COMPUTE_DTYPE="${BNB_4BIT_COMPUTE_DTYPE:-bfloat16}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DATA_DIR="${DATA_DIR:-../data}"
OUTPUT_DIR="${OUTPUT_DIR:-../checkpoints/qwen3_embedding}"
REPORT_TO="${REPORT_TO:-none}"

DEEPSPEED_ARGS=()
if [ -n "${DEEPSPEED:-}" ]; then
    DEEPSPEED_ARGS=(--deepspeed "$DEEPSPEED")
fi

echo "[$(date +%H:%M:%S)] Training $MODEL (train_type=$TRAIN_TYPE, gpus=$NPROC_PER_NODE)"

uv run python train.py \
    --model                  "$MODEL" \
    --train-type             "$TRAIN_TYPE" \
    --per-device-batch-size  "$PER_DEVICE_BATCH_SIZE" \
    --grad-accum             "$GRAD_ACCUM" \
    --epochs                 "$EPOCHS" \
    --lr                     "$LR" \
    --num-negatives          "$NUM_NEGATIVES" \
    --max-length             "$MAX_LENGTH" \
    --save-steps             "$SAVE_STEPS" \
    --save-total-limit       "$SAVE_TOTAL_LIMIT" \
    --quant-method           "$QUANT_METHOD" \
    --quant-bits             "$QUANT_BITS" \
    --bnb-4bit-compute-dtype "$BNB_4BIT_COMPUTE_DTYPE" \
    --nproc-per-node         "$NPROC_PER_NODE" \
    --data-dir               "$DATA_DIR" \
    --output-dir             "$OUTPUT_DIR" \
    --report-to              "$REPORT_TO" \
    "${DEEPSPEED_ARGS[@]}" \
    "$@"

echo "[$(date +%H:%M:%S)] Done."
if [ "$TRAIN_TYPE" = "lora" ]; then
    echo "Merge the adapters before evaluating (train.py printed the exact checkpoint path):"
    echo "  uv run swift export --adapters <checkpoint> --merge_lora true --use_hf true --output_dir $OUTPUT_DIR/merged"
fi
echo "Then evaluate with:"
echo "  cd ../shared && MODEL=$OUTPUT_DIR/merged bash evaluate.sh --pooling lasttoken --normalize"

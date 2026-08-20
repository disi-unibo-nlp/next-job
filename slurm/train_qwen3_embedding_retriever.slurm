#!/bin/bash
#SBATCH --job-name=jobhop-train-qwen3
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=96:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SCRATCH_DIR="${TMPDIR:-$HOME/next-job/tmp/${SLURM_JOB_ID:-manual}}"
export DOCKER_MEMORY="${DOCKER_MEMORY:-120g}"
mkdir -p logs handoff/checkpoints/qwen3_embedding "$SCRATCH_DIR"

bash run_docker.sh bash -lc '
  set -euo pipefail
  DATA_DIR="${DATA_DIR:-/workspace/handoff/data_full}"
  case "$DATA_DIR" in /*) ;; *) DATA_DIR="/workspace/$DATA_DIR" ;; esac
  python /workspace/scripts/check_handoff_training_data.py \
    --data-dir "$DATA_DIR" \
    --min-train-rows "${MIN_TRAIN_ROWS:-1000}"
  cd /workspace/handoff/04_qwen3_embedding
  uv sync
  DATA_DIR="$DATA_DIR" \
  OUTPUT_DIR="${OUTPUT_DIR:-/workspace/handoff/checkpoints/qwen3_embedding}" \
  MODEL="${MODEL:-Qwen/Qwen3-Embedding-8B}" \
  TRAIN_TYPE="${TRAIN_TYPE:-lora}" \
  PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}" \
  GRAD_ACCUM="${GRAD_ACCUM:-8}" \
  EPOCHS="${EPOCHS:-1}" \
  LR="${LR:-6e-6}" \
  NUM_NEGATIVES="${NUM_NEGATIVES:-7}" \
  MAX_LENGTH="${MAX_LENGTH:-512}" \
  SAVE_STEPS="${SAVE_STEPS:-4000}" \
  SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}" \
  QUANT_METHOD="${QUANT_METHOD:-bnb}" \
  QUANT_BITS="${QUANT_BITS:-4}" \
  BNB_4BIT_COMPUTE_DTYPE="${BNB_4BIT_COMPUTE_DTYPE:-bfloat16}" \
  bash -lc "bash run.sh ${TRAIN_ARGS:-}"
'

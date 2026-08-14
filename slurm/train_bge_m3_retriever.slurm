#!/bin/bash
#SBATCH --job-name=jobhop-train-bge-m3
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=48:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SCRATCH_DIR="${TMPDIR:-$HOME/next-job/tmp/${SLURM_JOB_ID:-manual}}"
export DOCKER_MEMORY="${DOCKER_MEMORY:-120g}"
mkdir -p logs handoff/checkpoints/bge_m3 "$SCRATCH_DIR"

bash run_docker.sh bash -lc '
  set -euo pipefail
  DATA_DIR="${DATA_DIR:-/workspace/handoff/data_full}"
  case "$DATA_DIR" in /*) ;; *) DATA_DIR="/workspace/$DATA_DIR" ;; esac
  python /workspace/scripts/check_handoff_training_data.py \
    --data-dir "$DATA_DIR" \
    --min-train-rows "${MIN_TRAIN_ROWS:-1000}"
  cd /workspace/handoff/03_bge_m3_flagembedding
  uv sync
  DATA_DIR="$DATA_DIR" \
  OUTPUT_DIR="${OUTPUT_DIR:-/workspace/handoff/checkpoints/bge_m3}" \
  MODEL="${MODEL:-BAAI/bge-m3}" \
  NEG_NUM="${NEG_NUM:-7}" \
  BATCH_SIZE="${BATCH_SIZE:-1}" \
  GRAD_ACCUM="${GRAD_ACCUM:-8}" \
  EPOCHS="${EPOCHS:-1}" \
  LR="${LR:-1e-5}" \
  PASSAGE_MAX_LEN="${PASSAGE_MAX_LEN:-512}" \
  QUERY_MAX_LEN="${QUERY_MAX_LEN:-128}" \
  bash -lc "bash run.sh --gradient-checkpointing ${TRAIN_ARGS:-}"
'

#!/bin/bash
#SBATCH --job-name=jobhop-train-st
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SCRATCH_DIR="${TMPDIR:-$HOME/next-job/tmp/${SLURM_JOB_ID:-manual}}"
export DOCKER_MEMORY="${DOCKER_MEMORY:-80g}"
mkdir -p logs handoff/checkpoints/sentence_transformer "$SCRATCH_DIR"

bash run_docker.sh bash -lc '
  set -euo pipefail
  cd /workspace/handoff/01_sentence_transformer
  uv sync
  DATA_DIR="${DATA_DIR:-/workspace/handoff/data}" \
  OUTPUT_DIR="${OUTPUT_DIR:-/workspace/handoff/checkpoints/sentence_transformer}" \
  MODEL="${MODEL:-sentence-transformers/all-MiniLM-L6-v2}" \
  LOSS="${LOSS:-mnrl}" \
  BATCH_SIZE="${BATCH_SIZE:-64}" \
  GRAD_ACCUM="${GRAD_ACCUM:-1}" \
  EPOCHS="${EPOCHS:-1}" \
  LR="${LR:-2e-5}" \
  NUM_NEGATIVES="${NUM_NEGATIVES:-4}" \
  bash -lc "bash run.sh ${TRAIN_ARGS:-}"
'

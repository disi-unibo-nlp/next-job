#!/bin/bash
#SBATCH --job-name=jobhop-train-lateon
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=36:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SCRATCH_DIR="${TMPDIR:-$HOME/next-job/tmp/${SLURM_JOB_ID:-manual}}"
export DOCKER_MEMORY="${DOCKER_MEMORY:-100g}"
mkdir -p logs handoff/checkpoints/late_interaction "$SCRATCH_DIR"

bash run_docker.sh bash -lc '
  set -euo pipefail
  cd /workspace/handoff/02_late_interaction
  uv sync
  DATA_DIR="${DATA_DIR:-/workspace/handoff/data}" \
  OUTPUT_DIR="${OUTPUT_DIR:-/workspace/handoff/checkpoints/late_interaction}" \
  MODEL="${MODEL:-lightonai/LateOn}" \
  LOSS="${LOSS:-contrastive}" \
  BATCH_SIZE="${BATCH_SIZE:-16}" \
  GRAD_ACCUM="${GRAD_ACCUM:-8}" \
  EPOCHS="${EPOCHS:-1}" \
  LR="${LR:-3e-6}" \
  NUM_NEGATIVES="${NUM_NEGATIVES:-4}" \
  DOCUMENT_LENGTH="${DOCUMENT_LENGTH:-300}" \
  QUERY_LENGTH="${QUERY_LENGTH:-32}" \
  bash -lc "bash run.sh ${TRAIN_ARGS:-}"
'

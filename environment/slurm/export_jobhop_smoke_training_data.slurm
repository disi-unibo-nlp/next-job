#!/bin/bash
#SBATCH --job-name=jobhop-export-smoke
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SCRATCH_DIR="${TMPDIR:-$HOME/next-job/tmp/${SLURM_JOB_ID:-manual}}"
export DOCKER_MEMORY="${DOCKER_MEMORY:-64g}"
export DOCKER_GPUS=none
mkdir -p logs handoff/data_smoke "$SCRATCH_DIR"

bash run_docker.sh python scripts/export_jobhop_training_data.py \
  --output-dir "${DATA_DIR:-handoff/data_smoke}" \
  --cache-dir "${CACHE_DIR:-/hf-cache}" \
  --random-seed "${RANDOM_SEED:-7}" \
  --max-train-samples "${MAX_TRAIN_SAMPLES:-500}" \
  --max-eval-cases "${MAX_EVAL_CASES:-200}" \
  --num-negatives "${NUM_NEGATIVES:-8}" \
  --train-example-mode "${TRAIN_EXAMPLE_MODE:-last-step}" \
  --overwrite

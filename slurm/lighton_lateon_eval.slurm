#!/bin/bash
#SBATCH --job-name=jobhop-lateon
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=18:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SCRATCH_DIR="${TMPDIR:-$HOME/next-job/tmp/${SLURM_JOB_ID:-manual}}"
export DOCKER_MEMORY="${DOCKER_MEMORY:-80g}"
mkdir -p logs outputs/lateon_eval "$SCRATCH_DIR"

bash run_docker.sh python scripts/lateon_eval.py \
  --model-name "lightonai/LateOn" \
  --max-eval-cases "${MAX_EVAL_CASES:-2000}" \
  --random-seed "${RANDOM_SEED:-7}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --doc-batch-size "${DOC_BATCH_SIZE:-32}" \
  --retrieve-k "${RETRIEVE_K:-200}" \
  --metric-ks "${METRIC_KS:-1,3,5,10,20,50,100}" \
  --output-dir "${OUTPUT_DIR:-outputs/lateon_eval}"

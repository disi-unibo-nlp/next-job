#!/bin/bash
#SBATCH --job-name=jobhop-nemotron3
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=24:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SCRATCH_DIR="${TMPDIR:-$HOME/next-job/tmp/${SLURM_JOB_ID:-manual}}"
export DOCKER_MEMORY="${DOCKER_MEMORY:-160g}"
mkdir -p logs outputs/dense_eval "$SCRATCH_DIR"

bash run_docker.sh python scripts/dense_eval.py \
  --model-name "nvidia/Nemotron-3-Embed-8B-BF16" \
  --max-eval-cases "${MAX_EVAL_CASES:-2000}" \
  --random-seed "${RANDOM_SEED:-7}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --retrieve-k "${RETRIEVE_K:-10}" \
  --metric-ks "${METRIC_KS:-1,3,5,10,20,50,100}" \
  --output-dir "${OUTPUT_DIR:-outputs/dense_eval}"

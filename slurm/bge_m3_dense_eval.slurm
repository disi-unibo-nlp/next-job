#!/bin/bash
#SBATCH --job-name=jobhop-bge-m3
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate next-job

cd "${SLURM_SUBMIT_DIR:-$PWD}"
SCRATCH_DIR="${TMPDIR:-$HOME/next-job/tmp/${SLURM_JOB_ID:-manual}}"
mkdir -p logs outputs/dense_eval "$SCRATCH_DIR"

export HF_HOME="${HF_CACHE_DIR:-$SCRATCH_DIR/hf-cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false

python scripts/dense_eval.py \
  --model-name "BAAI/bge-m3" \
  --max-eval-cases "${MAX_EVAL_CASES:-2000}" \
  --random-seed "${RANDOM_SEED:-7}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --retrieve-k "${RETRIEVE_K:-10}" \
  --metric-ks "${METRIC_KS:-1,3,5,10,20,50,100}" \
  --output-dir "${OUTPUT_DIR:-outputs/dense_eval}" \
  --cache-dir "$HF_HOME"

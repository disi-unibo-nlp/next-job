#!/bin/bash
#SBATCH --job-name=jobhop-lateon
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=18:00:00

set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate next-job

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs outputs/lateon_eval "${HF_CACHE_DIR:-/scratch/$USER/hf-cache}"

export HF_HOME="${HF_CACHE_DIR:-/scratch/$USER/hf-cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false

python scripts/lateon_eval.py \
  --model-name "lightonai/LateOn" \
  --max-eval-cases "${MAX_EVAL_CASES:-2000}" \
  --random-seed "${RANDOM_SEED:-7}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --doc-batch-size "${DOC_BATCH_SIZE:-32}" \
  --retrieve-k "${RETRIEVE_K:-200}" \
  --output-dir "${OUTPUT_DIR:-outputs/lateon_eval}" \
  --cache-dir "$HF_HOME"

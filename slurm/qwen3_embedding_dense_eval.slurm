#!/bin/bash
#SBATCH --job-name=jobhop-qwen3-emb
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs outputs/dense_eval "${HF_CACHE_DIR:-/scratch/$USER/hf-cache}"

export HF_HOME="${HF_CACHE_DIR:-/scratch/$USER/hf-cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false

python scripts/dense_eval.py \
  --model-name "Qwen/Qwen3-Embedding-0.6B" \
  --max-eval-cases "${MAX_EVAL_CASES:-2000}" \
  --random-seed "${RANDOM_SEED:-7}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --retrieve-k "${RETRIEVE_K:-10}" \
  --output-dir "${OUTPUT_DIR:-outputs/dense_eval}" \
  --cache-dir "$HF_HOME"

#!/bin/bash
set -euo pipefail

PHYS_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
IMAGE_NAME="${DOCKER_IMAGE:-jobhop-v2:latest}"
SCRATCH_DIR="${SCRATCH_DIR:-${TMPDIR:-$HOME/next-job/tmp/${SLURM_JOB_ID:-manual}}}"
CONTAINER_CACHE_DIR="${CONTAINER_CACHE_DIR:-/llms}"
LLM_CACHE_DIR="${LLM_CACHE_DIR:-/llms}"
DOCKER_MEMORY="${DOCKER_MEMORY:-80g}"

mkdir -p "$SCRATCH_DIR"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU_ARG="device=$CUDA_VISIBLE_DEVICES"
else
  GPU_ARG="all"
fi

docker_args=(
  run
  --rm
  -v "$PHYS_DIR":/workspace
  --memory="$DOCKER_MEMORY"
  --gpus "$GPU_ARG"
  -e TOKENIZERS_PARALLELISM=false
)

if [ -n "${HF_CACHE_DIR:-}" ]; then
  mkdir -p "$HF_CACHE_DIR"
  docker_args+=(
    -v "$HF_CACHE_DIR":/hf-cache
    -e HF_HOME=/hf-cache
    -e HF_HUB_CACHE=/hf-cache/hub
    -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub
    -e TRANSFORMERS_CACHE=/hf-cache/transformers
    -e HF_DATASETS_CACHE=/hf-cache/datasets
  )
  CONTAINER_CACHE_DIR=/hf-cache
elif [ -d "$LLM_CACHE_DIR" ]; then
  docker_args+=(
    -v "$LLM_CACHE_DIR":"$CONTAINER_CACHE_DIR"
    -e HF_HOME="$CONTAINER_CACHE_DIR"
    -e HF_HUB_CACHE="$CONTAINER_CACHE_DIR/hub"
    -e HUGGINGFACE_HUB_CACHE="$CONTAINER_CACHE_DIR/hub"
    -e TRANSFORMERS_CACHE="$CONTAINER_CACHE_DIR/transformers"
    -e HF_DATASETS_CACHE="$CONTAINER_CACHE_DIR/datasets"
  )
else
  mkdir -p "$SCRATCH_DIR/hf-cache"
  docker_args+=(
    -v "$SCRATCH_DIR/hf-cache":/hf-cache
    -e HF_HOME=/hf-cache
    -e HF_HUB_CACHE=/hf-cache/hub
    -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub
    -e TRANSFORMERS_CACHE=/hf-cache/transformers
    -e HF_DATASETS_CACHE=/hf-cache/datasets
  )
  CONTAINER_CACHE_DIR=/hf-cache
fi

docker_args+=(
  "$IMAGE_NAME"
  "$@"
)

exec docker "${docker_args[@]}"

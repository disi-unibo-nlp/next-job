#!/bin/bash
set -euo pipefail

PHYS_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
IMAGE_NAME="${DOCKER_IMAGE:-jobhop-v2:latest}"
SCRATCH_DIR="${SCRATCH_DIR:-${TMPDIR:-$HOME/next-job/tmp/${SLURM_JOB_ID:-manual}}}"
CONTAINER_CACHE_DIR="${CONTAINER_CACHE_DIR:-/hf-cache}"
DOCKER_MEMORY="${DOCKER_MEMORY:-80g}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$PHYS_DIR/hf-cache}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"

mkdir -p "$SCRATCH_DIR"

docker_args=(
  run
  --rm
  -v "$PHYS_DIR":/workspace
  --memory="$DOCKER_MEMORY"
  -e TOKENIZERS_PARALLELISM=false
  -e UV_LINK_MODE="$UV_LINK_MODE"
)

if [ -n "${DOCKER_NETWORK:-}" ]; then
  docker_args+=(--network "$DOCKER_NETWORK")
fi

for name in \
  MODEL LOSS BATCH_SIZE GRAD_ACCUM EPOCHS LR NUM_NEGATIVES NEG_NUM \
  DATA_DIR OUTPUT_DIR REPORT_TO DOCUMENT_LENGTH QUERY_LENGTH PASSAGE_MAX_LEN \
  QUERY_MAX_LEN TRAIN_TYPE PER_DEVICE_BATCH_SIZE MAX_LENGTH NPROC_PER_NODE \
  DEEPSPEED SAVE_STEPS TRAIN_ARGS ALLOW_SMOKE_DATA MIN_TRAIN_ROWS; do
  if [ -n "${!name:-}" ]; then
    docker_args+=(-e "$name=${!name}")
  fi
done

if [ "${DOCKER_GPUS:-auto}" != "none" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU_ARG="device=$CUDA_VISIBLE_DEVICES"
  else
    GPU_ARG="all"
  fi
  docker_args+=(--gpus "$GPU_ARG")
fi

mkdir -p "$HF_CACHE_DIR"
docker_args+=(
  -v "$HF_CACHE_DIR":"$CONTAINER_CACHE_DIR"
  -e HF_HOME="$CONTAINER_CACHE_DIR"
  -e HF_HUB_CACHE="$CONTAINER_CACHE_DIR/hub"
  -e HUGGINGFACE_HUB_CACHE="$CONTAINER_CACHE_DIR/hub"
  -e TRANSFORMERS_CACHE="$CONTAINER_CACHE_DIR/transformers"
  -e HF_DATASETS_CACHE="$CONTAINER_CACHE_DIR/datasets"
)

docker_args+=(
  "$IMAGE_NAME"
  "$@"
)

exec docker "${docker_args[@]}"

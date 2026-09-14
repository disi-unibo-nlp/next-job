#!/bin/bash
#SBATCH --job-name=jobhop-llm-rerank
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=168:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

RETRIEVER_MODEL="${RETRIEVER_MODEL:-handoff/checkpoints/bge_m3}"
SPLITS="${SPLITS:-validation test}"
MAX_EVAL_CASES="${MAX_EVAL_CASES:-none}"
RANDOM_SEED="${RANDOM_SEED:-7}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CANDIDATE_K="${CANDIDATE_K:-100}"
RETURN_K="${RETURN_K:-20}"
METRIC_KS="${METRIC_KS:-1,3,5,10,20,50,100}"
CANDIDATE_DESCRIPTION_CHARS="${CANDIDATE_DESCRIPTION_CHARS:-160}"

LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3.5-9B}"
LLM_SERVED_MODEL_NAME="${LLM_SERVED_MODEL_NAME:-$LLM_MODEL}"
LLM_OUTPUT_DIR="${LLM_OUTPUT_DIR:-outputs/llm_wiki_rerank_qwen35}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

export SCRATCH_DIR="${TMPDIR:-$HOME/next-job/tmp/${SLURM_JOB_ID:-manual}}"
export DOCKER_MEMORY="${DOCKER_MEMORY:-120g}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-$PWD/hf-cache}"
mkdir -p "$SCRATCH_DIR" "$HF_CACHE_DIR" logs outputs/llm_candidate_cache

RETRIEVER_SLUG="$(printf "%s" "$RETRIEVER_MODEL" | sed 's#[^A-Za-z0-9._-]#_#g')"
VLLM_CONTAINER="jobhop-vllm-${SLURM_JOB_ID:-manual}"
VLLM_LOG="logs/vllm-${SLURM_JOB_ID:-manual}.log"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU_ARG="device=$CUDA_VISIBLE_DEVICES"
else
  GPU_ARG="all"
fi

cleanup() {
  docker rm -f "$VLLM_CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[$(date --iso-8601=seconds)] JobHop LLM rerank job"
echo "Retriever: $RETRIEVER_MODEL"
echo "Splits: $SPLITS"
echo "LLM: $LLM_MODEL"
echo "vLLM image: $VLLM_IMAGE"

echo "[$(date --iso-8601=seconds)] Step 1/3: generate retriever candidate caches"
for SPLIT in $SPLITS; do
  CANDIDATE_CACHE="outputs/llm_candidate_cache/${RETRIEVER_SLUG}/${SPLIT}_top${CANDIDATE_K}_seed${RANDOM_SEED}.jsonl"
  echo "[$(date --iso-8601=seconds)] Candidate cache: $CANDIDATE_CACHE"
  bash run_docker.sh python scripts/jobhop_llm_wiki_rerank.py \
    --retriever-model "$RETRIEVER_MODEL" \
    --split "$SPLIT" \
    --max-eval-cases "$MAX_EVAL_CASES" \
    --random-seed "$RANDOM_SEED" \
    --batch-size "$BATCH_SIZE" \
    --candidate-k "$CANDIDATE_K" \
    --return-k "$RETURN_K" \
    --metric-ks "$METRIC_KS" \
    --candidate-cache "$CANDIDATE_CACHE" \
    --save-candidates-only \
    --output-dir "$LLM_OUTPUT_DIR" \
    --cache-dir /hf-cache \
    --require-cuda
done

echo "[$(date --iso-8601=seconds)] Step 2/3: start vLLM server"
cleanup

VLLM_CMD=(
  docker run
  -d
  --rm
  --name "$VLLM_CONTAINER"
  --network host
  --ipc host
  --gpus "$GPU_ARG"
  -v "$HF_CACHE_DIR":/hf-cache
  -e HF_HOME=/hf-cache
  -e HF_HUB_CACHE=/hf-cache/hub
  -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub
  "$VLLM_IMAGE"
  --model "$LLM_MODEL"
  --served-model-name "$LLM_SERVED_MODEL_NAME"
  --host 0.0.0.0
  --port "$VLLM_PORT"
  --trust-remote-code
  --max-model-len "$VLLM_MAX_MODEL_LEN"
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$VLLM_MAX_NUM_SEQS"
)

if [ -n "$VLLM_EXTRA_ARGS" ]; then
  read -r -a EXTRA_ARGS <<< "$VLLM_EXTRA_ARGS"
  VLLM_CMD+=("${EXTRA_ARGS[@]}")
fi

"${VLLM_CMD[@]}"
docker logs -f "$VLLM_CONTAINER" > "$VLLM_LOG" 2>&1 &

echo "[$(date --iso-8601=seconds)] Waiting for vLLM at http://127.0.0.1:${VLLM_PORT}/v1/models"
READY=0
for _ in $(seq 1 180); do
  if curl -fs "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
    READY=1
    break
  fi
  sleep 10
done

if [ "$READY" -ne 1 ]; then
  echo "vLLM did not become ready. Last vLLM log lines:"
  tail -100 "$VLLM_LOG" || true
  exit 1
fi

echo "[$(date --iso-8601=seconds)] Step 3/3: run LLM reranking from cached candidates"
for SPLIT in $SPLITS; do
  CANDIDATE_CACHE="outputs/llm_candidate_cache/${RETRIEVER_SLUG}/${SPLIT}_top${CANDIDATE_K}_seed${RANDOM_SEED}.jsonl"
  DOCKER_NETWORK=host DOCKER_GPUS=none bash run_docker.sh python scripts/jobhop_llm_wiki_rerank.py \
    --retriever-model "$RETRIEVER_MODEL" \
    --split "$SPLIT" \
    --max-eval-cases "$MAX_EVAL_CASES" \
    --random-seed "$RANDOM_SEED" \
    --candidate-k "$CANDIDATE_K" \
    --return-k "$RETURN_K" \
    --metric-ks "$METRIC_KS" \
    --candidate-cache "$CANDIDATE_CACHE" \
    --rerank-from-candidates \
    --llm-base-url "http://127.0.0.1:${VLLM_PORT}/v1" \
    --llm-model "$LLM_SERVED_MODEL_NAME" \
    --max-tokens "${LLM_MAX_TOKENS:-1024}" \
    --timeout "${LLM_TIMEOUT:-180}" \
    --candidate-description-chars "$CANDIDATE_DESCRIPTION_CHARS" \
    --output-dir "$LLM_OUTPUT_DIR" \
    --cache-dir /hf-cache
done

echo "[$(date --iso-8601=seconds)] Done."

# JobHop LLM-Wiki-Style Reranking

This is a JobHop-specific adaptation of the LLM_Wiki local LLM pattern. It does not use the Wiki QA task, Wiki metrics, Milvus, or SQuAD/MultiHop data.

The script keeps the JobHop protocol:

- last-step next-occupation prediction
- train-split ESCO candidate pool
- validation/test split from JobHop
- MRR and Recall/Accuracy/Precision/NDCG/MAP at configurable K values

The LLM step follows the suggested option 2:

1. Build top-k candidates with a retriever model.
2. Send the full candidate list to a local OpenAI-compatible LLM endpoint.
3. Ask the LLM to return the most relevant candidate ids.
4. Use the returned ids as a reranking signal.

## Prompt-Only Smoke Test

This writes the candidate prompt and baseline retriever metrics without calling the LLM:

```bash
python scripts/jobhop_llm_wiki_rerank.py \
  --retriever-model BAAI/bge-m3 \
  --split validation \
  --max-eval-cases 1 \
  --candidate-k 100 \
  --return-k 20 \
  --dry-run \
  --output-dir outputs/llm_wiki_rerank_smoke \
  --cache-dir hf-cache
```

## Local LLM Run

Start a local vLLM/OpenAI-compatible server first, then run:

```bash
python scripts/jobhop_llm_wiki_rerank.py \
  --retriever-model BAAI/bge-m3 \
  --split validation \
  --max-eval-cases 20 \
  --candidate-k 100 \
  --return-k 20 \
  --llm-base-url http://localhost:8000/v1 \
  --llm-model Qwen/Qwen3.5-9B \
  --output-dir outputs/llm_wiki_rerank \
  --cache-dir hf-cache
```

Use `--max-eval-cases none` only after the small run is verified.

## SLURM vLLM Reranking Job

Use `slurm/llm_rerank_eval.slurm` for cluster runs. This is one SLURM job and one GPU allocation. Inside the job it:

1. Generates retriever top-k candidate caches.
2. Starts a `Qwen/Qwen3.5-9B` vLLM OpenAI-compatible server.
3. Runs LLM reranking from the cached candidates.

Example:

```bash
RETRIEVER_MODEL=handoff/checkpoints/bge_m3 \
SPLITS="validation test" \
MAX_EVAL_CASES=none \
sbatch -N 1 -w faretra --gpus=nvidia_geforce_rtx_3090:1 \
  slurm/llm_rerank_eval.slurm
```

This avoids running a separate interactive LLM server and prevents the previous `localhost:8000 Connection refused` failure mode.

## Outputs

The run directory contains:

- `requests.jsonl`
- `responses.jsonl`
- `parsed_results.jsonl`
- `failures.jsonl`
- `base_retriever_metrics.*`
- `metrics.*`
- `run_config.*`
- `runtime.*`
- `dataset_summary.*`
- `preprocessing_config.*`

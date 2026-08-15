# JobHop Retriever Training Handoff

This repository now includes the retriever-training handoff under `handoff/`.
The handoff recipes are kept as training templates for four retriever families:

- `handoff/01_sentence_transformer`: dense bi-encoder training with Sentence Transformers.
- `handoff/02_late_interaction`: ColBERT/LateOn-style late interaction training with PyLate.
- `handoff/03_bge_m3_flagembedding`: BGE-M3 training with FlagEmbedding.
- `handoff/04_qwen3_embedding`: Qwen3-Embedding training with ms-swift.

## Important Protocol Notes

The handoff originally prepares `sentence-transformers/natural-questions`.
Do not use that default data preparation for JobHop experiments.

For JobHop, use `scripts/export_jobhop_training_data.py`. It reuses the existing
JobHop v2 preprocessing from `scripts/jobhop_eval_common.py`:

- `aida-ugent/JobHop` split loading.
- `danieldux/ESCO` occupation metadata loading.
- unknown-code removal.
- consecutive duplicate collapse.
- chronological trajectory ordering.
- train-only candidate occupation pool.
- validation/test filtering to targets seen in the train candidate pool.
- existing query and candidate text templates.

The evaluation protocol remains last-step next-occupation prediction. The
exporter default `--train-example-mode last-step` is the most conservative
training-data version of that protocol. `--train-example-mode all-prefixes` is
available only as an explicit training expansion and should be reported if used.

## Export JobHop Training Data

On the cluster, after building the Docker image:

Full-data export writes to `handoff/data_full` by default:

```bash
sbatch slurm/export_jobhop_training_data.slurm
```

For a small smoke export, use the separate smoke script. It writes to
`handoff/data_smoke` so it cannot overwrite the full-data files:

```bash
sbatch slurm/export_jobhop_smoke_training_data.slurm
```

The full exporter writes:

- `handoff/data_full/train.jsonl`
- `handoff/data_full/eval_corpus.jsonl`
- `handoff/data_full/eval_queries.jsonl` defaults to validation queries for training-time evaluation
- `handoff/data_full/eval_queries_validation.jsonl`
- `handoff/data_full/eval_queries_test.jsonl`
- `handoff/data_full/export_summary.json`
- `handoff/data_full/export_summary.csv`
- `handoff/data_full/export_failure.json` if export fails

Each training row has the canonical handoff format:

```json
{"query": "...", "positive": "...", "negatives": ["...", "..."]}
```

## Train One Retriever

The SLURM defaults are the formal training choices:

- SentenceTransformer dense: `BAAI/bge-base-en-v1.5`
- Late interaction: `lightonai/LateOn`
- BGE-M3: `BAAI/bge-m3`
- Qwen embedding: `Qwen/Qwen3-Embedding-8B`

The smaller handoff defaults such as `sentence-transformers/all-MiniLM-L6-v2`
and `Qwen/Qwen3-Embedding-0.6B` should be treated as smoke/debug choices, not as
the main experimental models.

Start a formal dense run with:

```bash
sbatch slurm/train_sentence_transformer_retriever.slurm
```

Useful smoke run:

```bash
sbatch slurm/export_jobhop_smoke_training_data.slurm
DATA_DIR=handoff/data_smoke ALLOW_SMOKE_DATA=1 MODEL=sentence-transformers/all-MiniLM-L6-v2 \
  TRAIN_ARGS="--max-steps 5 --no-save --no-eval" \
  sbatch slurm/train_sentence_transformer_retriever.slurm
```

Other recipes:

```bash
sbatch slurm/train_lateon_retriever.slurm
sbatch slurm/train_bge_m3_retriever.slurm
sbatch slurm/train_qwen3_embedding_retriever.slurm
```

Each SLURM script runs one model family only and does not use conda or hard-code
a node.

Training scripts read `handoff/data_full` by default. Before training starts,
they check the number of rows in `train.jsonl`. If the file looks like a small
smoke export, training stops unless `ALLOW_SMOKE_DATA=1` is set deliberately.

BGE-M3 and Qwen3 default to `SAVE_STEPS=50000` for full runs. This is deliberate:
the full JobHop run has about 32k optimizer steps, and frequent intermediate
checkpoints can exceed the 150 GB user quota. Override `SAVE_STEPS` only when
you explicitly want intermediate checkpoints and have enough disk space.

## Docker / CUDA Notes

`build/Dockerfile` installs `uv` and keeps the existing PyTorch CUDA checks:

- `torch==2.7.1+cu128`
- `torchvision==0.22.1+cu128`
- `torchaudio==2.7.1+cu128`

The handoff `pyproject.toml` files now pin `torch==2.7.1+cu128` from the
PyTorch CUDA 12.8 wheel index. If a future dependency requires a different
torch, `uv sync` should fail instead of silently upgrading PyTorch.

`run_docker.sh` now defaults Hugging Face cache writes to a user-writable
directory under the project (`hf-cache`) instead of `/llms`. Pass
`HF_CACHE_DIR=/some/writable/path` if a different writable cache should be used.

## Final Evaluation

The handoff evaluator is useful for training sanity checks, but the final
JobHop numbers should still use the existing evaluation scripts:

- `scripts/dense_eval.py`
- `scripts/lateon_eval.py`

This preserves validation-only alpha selection and one final test evaluation.

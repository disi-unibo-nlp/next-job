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

```bash
sbatch slurm/export_jobhop_training_data.slurm
```

For a small smoke export:

```bash
MAX_TRAIN_SAMPLES=500 MAX_EVAL_CASES=200 sbatch slurm/export_jobhop_training_data.slurm
```

The exporter writes:

- `handoff/data/train.jsonl`
- `handoff/data/eval_corpus.jsonl`
- `handoff/data/eval_queries.jsonl` defaults to validation queries for training-time evaluation
- `handoff/data/eval_queries_validation.jsonl`
- `handoff/data/eval_queries_test.jsonl`
- `handoff/data/export_summary.json`
- `handoff/data/export_summary.csv`
- `handoff/data/export_failure.json` if export fails

Each training row has the canonical handoff format:

```json
{"query": "...", "positive": "...", "negatives": ["...", "..."]}
```

## Train One Retriever

Start with the dense recipe as a smoke test:

```bash
sbatch slurm/train_sentence_transformer_retriever.slurm
```

Useful smoke run:

```bash
MAX_TRAIN_SAMPLES=500 MAX_EVAL_CASES=200 sbatch slurm/export_jobhop_training_data.slurm
TRAIN_ARGS="--max-steps 5 --no-save --no-eval" sbatch slurm/train_sentence_transformer_retriever.slurm
```

Other recipes:

```bash
sbatch slurm/train_lateon_retriever.slurm
sbatch slurm/train_bge_m3_retriever.slurm
sbatch slurm/train_qwen3_embedding_retriever.slurm
```

Each SLURM script runs one model family only and does not use conda or hard-code
a node.

## Docker / CUDA Notes

`build/Dockerfile` installs `uv` and keeps the existing PyTorch CUDA checks:

- `torch==2.7.1+cu128`
- `torchvision==0.22.1+cu128`
- `torchaudio==2.7.1+cu128`

The handoff `pyproject.toml` files now pin `torch==2.7.1+cu128` from the
PyTorch CUDA 12.8 wheel index. If a future dependency requires a different
torch, `uv sync` should fail instead of silently upgrading PyTorch.

## Final Evaluation

The handoff evaluator is useful for training sanity checks, but the final
JobHop numbers should still use the existing evaluation scripts:

- `scripts/dense_eval.py`
- `scripts/lateon_eval.py`

This preserves validation-only alpha selection and one final test evaluation.

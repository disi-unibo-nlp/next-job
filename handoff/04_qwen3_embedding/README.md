# 04 — Qwen3-Embedding (ms-swift)

Qwen3-Embedding is a **decoder LLM repurposed as an embedder**. Text runs
through a causal transformer and the hidden state of the **last token** becomes
the embedding. Everything downstream follows from that:

- Pooling is `lasttoken`, not `mean` or `cls`. Get this wrong at evaluation time
  and the numbers collapse — see [Evaluating](#evaluating).
- Queries carry a natural-language **instruction**, because the backbone was
  instruction-tuned. Documents do not.
- Even the 0.6B variant is larger than every other recipe here, so LoRA is the
  practical default on one GPU.

Training follows the upstream recipe from
[QwenLM/Qwen3-Embedding `docs/training/SWIFT.md`](https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md),
which uses **ms-swift**, not Sentence Transformers. `train.py` converts the
shared data into swift's format and then builds and runs the `swift sft`
command line, so the whole invocation stays visible and reproducible.

## Setup

```bash
cd handoff/04_qwen3_embedding
uv sync
```

Requires `../data/train.jsonl` — build it first with `../shared/prepare_data.sh`.

`ms-swift` is pinned to `4.4.2` in `pyproject.toml`. Its `--task_type embedding`
surface, `--loss_type` names and `INFONCE_*` environment variables move between
releases, and `train.py` builds a command line against that surface. Bump the
pin deliberately and re-check `build_swift_command()` / `build_swift_env()`.

## Run

```bash
bash run.sh --dry-run          # converts the data, prints the swift command, runs nothing
bash run.sh                    # LoRA on one GPU
TRAIN_TYPE=full DEEPSPEED=zero3 NPROC_PER_NODE=8 bash run.sh    # upstream full-FT recipe
```

`--dry-run` prints the exact `swift sft ...` line and the environment it would
run under. Read it before the first real run; it is the fastest way to learn
what this recipe actually does.

## How the data is used

swift models every text as a chat message list. `train.py` writes
`../checkpoints/qwen3_embedding/swift_data/train.jsonl`:

```json
{"messages": [{"role": "user", "content": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: who wrote ecclesiastes"}],
 "positive_messages": [[{"role": "user", "content": "Ecclesiastes is one of 24 books ..."}]],
 "negative_messages": [[{"role": "user", "content": "Book of Proverbs ..."}],
                       [{"role": "user", "content": "Song of Songs ..."}]]}
```

Two things catch people out:

1. **The double nesting.** `positive_messages` and `negative_messages` are lists
   *of message lists*, not lists of messages — swift supports several positives
   per query, each its own conversation.
2. **The instruction goes inside the query content**, not in a separate field.
   `--query-instruction` sets it, defaulting to Qwen3-Embedding's published
   retrieval prompt. Documents are never prefixed. Whatever you train with must
   be passed to the evaluator as `--query-prompt`.

## Choosing a loss

| `--loss-type` | usable here | notes |
|---|---|---|
| `infonce` *(default)* | yes | Contrastive over the positive and negatives, plus in-batch negatives. This is what the upstream docs recommend. |
| `cosine_similarity` | no | Needs a float `label` per pair. The shared triplet data carries no labels, so `train.py` refuses rather than training on a broken objective. |

InfoNCE is configured through **environment variables**, not command-line flags
— a swift convention worth knowing before you go looking for a `--temperature`
that does not exist. `train.py` sets them from its own flags:

| env var | set from | default | meaning |
|---|---|---|---|
| `INFONCE_TEMPERATURE` | `--infonce-temperature` | 0.01 | Much sharper than the other recipes. |
| `INFONCE_USE_BATCH` | `--infonce-use-batch` / `--no-infonce-use-batch` | true | Use other rows in the batch as extra negatives. |
| `INFONCE_HARD_NEGATIVES` | `--num-negatives` | 7 | Explicit negatives per query. |
| `INFONCE_MASK_FAKE_NEGATIVE` | `--infonce-mask-fake-negative` | false | Drop in-batch negatives that look like false negatives. Worth turning on if in-batch negatives are noisy. |
| `MAX_POSITIVE_SAMPLES` | fixed | 1 | This pipeline always writes one positive. |
| `MAX_NEGATIVE_SAMPLES` | `--num-negatives` | 7 | |

### Effective batch size

The number that actually matters is not `--per-device-batch-size`:

```
texts per device per step = per_device_train_batch_size
                          x MAX_POSITIVE_SAMPLES
                          x (1 + MAX_NEGATIVE_SAMPLES)
```

At the defaults that is `4 x 1 x 8 = 32` texts of up to `--max-length` tokens
through an LLM, per device, per step. Raising `--num-negatives` costs memory as
fast as raising the batch size. `train.py` prints this figure before launching.

## Parameters

### Model

| flag | default | notes |
|---|---|---|
| `--model` | `Qwen/Qwen3-Embedding-0.6B` | Also `-4B` and `-8B`. Start at 0.6B. |
| `--model-type` | `qwen3_emb` | swift's registry key for this family. Leave it. |
| `--train-type` | `lora` | `full` matches the upstream example but wants ZeRO-3 and several GPUs. |
| `--lora-rank` / `--lora-alpha` | 8 / 32 | |
| `--target-modules` | `all-linear` | swift shorthand for every linear layer. |
| `--max-length` | 512 | Queries and passages share this budget. |

### Optimisation

| flag | default | notes |
|---|---|---|
| `--per-device-batch-size` | 4 | See the effective-batch formula above. |
| `--grad-accum` | 8 | |
| `--lr` | 6e-6 | The upstream value. LoRA tolerates more; full fine-tuning does not. |
| `--epochs` | 1 | Upstream uses 5 on a much smaller dataset. |
| `--gradient-checkpointing` | on | `--no-gradient-checkpointing` to disable. |
| `--deepspeed` | none | `zero2`, `zero3` or `zero3-offload`. Required for `--train-type full`. |
| `--nproc-per-node` | 1 | Above 1 sets `NPROC_PER_NODE`, which is how swift decides to launch distributed. |
| `--max-steps` | none | Small value for smoke runs. |

### Escape hatch

`--convert-only` writes the swift JSONL and stops, so you can run `swift sft`
by hand with flags this script does not expose:

```bash
uv run python train.py --convert-only
uv run swift sft --model Qwen/Qwen3-Embedding-0.6B --task_type embedding \
    --model_type qwen3_emb --loss_type infonce \
    --dataset ../checkpoints/qwen3_embedding/swift_data/train.jsonl ...
```

## Merging LoRA adapters

A LoRA run saves adapters, not a usable encoder. Merge before evaluating:

swift versions each run as `<output_dir>/v<N>-<timestamp>/checkpoint-<step>`, so
the path is not predictable — `train.py` prints the exact merge command with the
real checkpoint path when the run finishes:

```bash
uv run swift export \
    --adapters ../checkpoints/qwen3_embedding/v0-20260812-161928/checkpoint-500 \
    --merge_lora true \
    --use_hf true \
    --output_dir ../checkpoints/qwen3_embedding/merged
```

`--use_hf true` matters: without it `swift export` resolves the base model
through ModelScope and fails if only the HuggingFace copy is cached.

`--train-type full` skips this step entirely.

## Evaluating

```bash
cd ../shared
uv run python evaluate.py --backend dense \
    --model-path ../checkpoints/qwen3_embedding/merged \
    --pooling lasttoken --normalize \
    --query-prompt "Instruct: Given a web search query, retrieve relevant passages that answer the query
Query: "
```

**`--pooling lasttoken` is not optional.** swift writes a plain transformers
directory — no `modules.json`, no `1_Pooling/` — so `SentenceTransformer(path)`
silently falls back to mean pooling. On a 358-document smoke corpus that single
mistake cost **7.8 NDCG@10 points** (93.4 with `lasttoken`, 85.7 with `mean`),
and the gap widens on a realistic corpus. It looks like a failed training run
rather than a configuration error, which is what makes it expensive.

Pass the same `--query-prompt` you trained with, for the same reason.

## Reference

- [Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding) and its
  [SWIFT training doc](https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md)
- [ms-swift](https://github.com/modelscope/ms-swift) —
  [embedding training guide](https://swift.readthedocs.io/en/latest/BestPractices/Embedding.html)
- ms-swift also trains **rerankers** (`--task_type generative_reranker`), which
  pairs naturally with any of these retrievers as a second stage. Out of scope
  here, but the data format is a sibling of the one above.

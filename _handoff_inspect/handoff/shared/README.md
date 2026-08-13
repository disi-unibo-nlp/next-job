# Shared: dataset preparation and evaluation

Two scripts that every recipe in this handoff depends on. Run `prepare_data.py`
once before any training, and `evaluate.py` before and after each run.

## Setup

```bash
cd handoff/shared
uv sync                              # base environment
uv sync --extra late-interaction     # add PyLate, needed to evaluate recipe 02
uv sync --extra faiss                # add faiss-cpu for --use-faiss mining
```

## Step 1 — build the dataset

```bash
bash prepare_data.sh
# or, with explicit flags:
uv run python prepare_data.py --train-size 50000 --eval-queries 2000 --num-negatives 8
```

Source: [`sentence-transformers/natural-questions`](https://huggingface.co/datasets/sentence-transformers/natural-questions)
— 100k real Google search queries paired with the Wikipedia passage that answers
them. One `train` split, columns `query` and `answer`.

The script shuffles with `--seed`, peels off `--eval-queries` rows for
evaluation, mines hard negatives over the rest, and writes four files into
`--output-dir` (default `../data/`, git-ignored).

### Output schema

`train.jsonl` — one row per training example:

```json
{"query": "who wrote the book of ecclesiastes",
 "positive": "Ecclesiastes Ecclesiastes is one of 24 books ...",
 "negatives": ["Book of Proverbs The Book of Proverbs ...", "..."]}
```

With `--with-scores` two extra fields appear, carrying the mining model's
similarity scores: `"positive_score": 0.82` and `"negative_scores": [0.61, ...]`.
Only knowledge-distillation losses need them.

`eval_corpus.jsonl` — the passages to retrieve from:

```json
{"docid": "d0", "text": "Ecclesiastes Ecclesiastes is one of 24 books ..."}
```

`eval_queries.jsonl` — the held-out queries and their gold documents:

```json
{"qid": "q0", "query": "who wrote the book of ecclesiastes", "positive_docids": ["d0"]}
```

`summary.json` — what was built and with which mining settings.

### Hard-negative mining

Negatives come from `sentence_transformers.util.mine_hard_negatives`. It
encodes every training passage with `--mining-model`, retrieves the nearest
passages for each query, and then throws away the ones that are probably
relevant after all — that filtering is the whole point, because a "negative"
that actually answers the query teaches the model the wrong thing.

| flag | default | what it does |
|---|---|---|
| `--num-negatives` | 8 | Negatives per query. Rows where fewer survive filtering are dropped entirely. |
| `--range-min` | 10 | Skip the top-10 nearest passages; those are the likeliest false negatives. |
| `--range-max` | 50 | Only consider down to rank 50; below that negatives get too easy. |
| `--max-score` | 0.8 | Discard any candidate scoring above 0.8 — too similar to be a negative. |
| `--relative-margin` | 0.05 | Keep only candidates scoring at most 95% of the positive's score. |
| `--absolute-margin` | off | Alternative to the above: require `pos - neg >= margin`. |
| `--sampling-strategy` | `top` | `top` keeps the hardest surviving candidates; `random` samples from the range for a gentler curriculum. |
| `--mining-model` | `all-MiniLM-L6-v2` | Any SentenceTransformer. A stronger miner gives harder, cleaner negatives at the cost of mining time. |
| `--use-faiss` | off | Approximate search instead of an exact matmul. Worth it above ~100k passages; needs the `faiss` extra. |

Harder negatives are not automatically better. If `--max-score` is too loose you
mine false negatives and the model learns to push apart passages that both
answer the query; too tight and every negative is trivially wrong and the loss
goes to zero. The defaults follow the Sentence Transformers guidance for
Natural Questions.

### Sizing

`--train-size 50000` and `--eval-queries 2000` are a sensible default on one
GPU. For a smoke test:

```bash
uv run python prepare_data.py --train-size 200 --eval-queries 50 --eval-corpus-size 500 \
    --num-negatives 4 --overwrite
```

Rebuilding is skipped when the output files already exist; pass `--overwrite` to
force it.

`--eval-corpus-size` controls how many distractor passages sit alongside the
gold ones. Every eval positive is always kept regardless of that cap, so every
query stays answerable; a bigger corpus just makes the task harder.

## Step 2 — evaluate

```bash
bash evaluate.sh                                        # needs MODEL=...
MODEL=sentence-transformers/all-MiniLM-L6-v2 bash evaluate.sh
uv run python evaluate.py --backend dense --model-path ../checkpoints/sentence_transformer/final
```

Metrics come from `sentence_transformers.evaluation.InformationRetrievalEvaluator`:
accuracy, precision, recall, NDCG, MRR and MAP at each `--k` (default 1, 3, 5, 10).
NDCG@10 is the number to quote.

### Backends

| `--backend` | recipes | encoder | scoring |
|---|---|---|---|
| `dense` | 01, 03, 04 | `SentenceTransformer` | cosine |
| `late-interaction` | 02 | PyLate `ColBERT` | MaxSim |

The `late-interaction` path wraps the PyLate model so it looks like a dense
encoder to the evaluator: `encode_query` / `encode_document` return token
embeddings zero-padded into a `(batch, tokens, dim)` tensor, and MaxSim is
injected through the evaluator's `score_functions` hook. Padding does not
distort the score because MaxSim maxes over document tokens and the padded rows
are all zero. If scoring runs out of memory, lower `--score-query-chunk` and
`--score-doc-chunk`; they control the size of the `(q, d, q_tokens, d_tokens)`
intermediate, which is what dominates memory.

### Checkpoints without a SentenceTransformer config

`SentenceTransformer(path)` falls back to **mean pooling** when a directory
holds only transformers files. That is silently wrong for models trained with a
different pooling head. Pass `--pooling` to rebuild the module stack explicitly:

| checkpoint | flags |
|---|---|
| recipe 01, recipe 02 | none — the saved config is correct |
| recipe 03 (BGE-M3) | `--pooling cls --normalize` — **required**; FlagEmbedding writes no `modules.json` |
| recipe 04 (Qwen3-Embedding) | `--pooling lasttoken --normalize` — **required**; swift writes no `modules.json` |

Instruction-tuned models also want their prompt at eval time, e.g. for
Qwen3-Embedding:

```bash
uv run python evaluate.py --backend dense --model-path ../checkpoints/qwen3_embedding/merged \
    --pooling lasttoken --normalize \
    --query-prompt "Instruct: Given a web search query, retrieve relevant passages that answer the query
Query: "
```

The prompt must match what the model was trained with, otherwise the numbers
understate it.

## Reference

- `--max-queries N` evaluates only the first N queries — useful for checking the
  plumbing without waiting for a full pass.
- `--output path.json` writes the full metric dict; `evaluate.sh` does this
  automatically into `../results/`.

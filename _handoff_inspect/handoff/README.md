# Retriever training handoff

Four ways to train a text retriever, each in its own folder, each self-contained
and each trained on the **same dataset** so the results are directly comparable.
Every folder holds a Python trainer, a bash launcher, a `uv` environment and a
README that explains the parameters that matter for that method.

Nothing here imports from the research code in `../src`. Copy a folder somewhere
else and it still runs.

## Where to find what

| # | folder | method | scoring | start here if |
|---|---|---|---|---|
| — | [`shared/`](shared/) | dataset preparation + evaluation | — | always: run this first and last |
| 1 | [`01_sentence_transformer/`](01_sentence_transformer/) | dense bi-encoder (Sentence Transformers) | cosine over one vector | you are new to this — read this one first |
| 2 | [`02_late_interaction/`](02_late_interaction/) | ColBERT-style multi-vector (PyLate) | MaxSim over per-token vectors | recall matters more than index size |
| 3 | [`03_bge_m3_flagembedding/`](03_bge_m3_flagembedding/) | BGE-M3 (FlagEmbedding) | dense + sparse + ColBERT heads at once | you want lexical and semantic matching together, or multilingual |
| 4 | [`04_qwen3_embedding/`](04_qwen3_embedding/) | Qwen3-Embedding (ms-swift) | cosine over an LLM's last-token state | you want the strongest model and have the GPUs |

Ordered by cost. Recipe 1 trains a 22M model in minutes; recipe 4 fine-tunes a
0.6B–8B LLM.

## The three steps

Every method follows the same shape.

**1. Build the dataset (once).**

```bash
cd shared
uv sync
bash prepare_data.sh
```

Downloads [`sentence-transformers/natural-questions`](https://huggingface.co/datasets/sentence-transformers/natural-questions)
(100k real search queries paired with the Wikipedia passage that answers them),
holds out an evaluation slice, mines hard negatives, and writes `data/`.

**2. Get the baseline.**

```bash
MODEL=sentence-transformers/all-MiniLM-L6-v2 bash evaluate.sh
```

Score the base model *before* training. Without this number, an after-training
number means nothing.

**3. Train, then re-evaluate.**

```bash
cd ../01_sentence_transformer
uv sync
bash run.sh --dry-run       # resolve everything, train nothing
bash run.sh

cd ../shared
MODEL=../checkpoints/sentence_transformer/final bash evaluate.sh
```

Every trainer accepts `--dry-run` (resolve config and data, print the plan, exit)
and `--max-steps N` (stop after N optimizer steps). Use both before committing to
a long run.

Each recipe's defaults are tuned for that method, not for cross-method parity —
the light models default to fewer hard negatives and much larger batches (so more
in-batch negatives), the heavy ones the reverse. For a strict comparison, fix
`NUM_NEGATIVES` to the same value everywhere and say so when reporting.

## The shared data format

`shared/prepare_data.py` writes one canonical format; each trainer converts it to
whatever its framework wants, in-process. That conversion is documented in each
folder's README — it is the part worth understanding, because it is where a
retrieval dataset meets a specific loss.

```
data/train.jsonl        {"query": str, "positive": str, "negatives": [str, ...]}
data/eval_corpus.jsonl  {"docid": "d0", "text": str}
data/eval_queries.jsonl {"qid": "q0", "query": str, "positive_docids": ["d0"]}
```

To train on your own data, produce these three files and skip `prepare_data.py`
entirely — nothing downstream knows where they came from.

`data/`, `checkpoints/` and `results/` are all git-ignored.

## Evaluation

One script scores every method: `shared/evaluate.py`, built on
`sentence_transformers.evaluation.InformationRetrievalEvaluator`. It reports
accuracy, precision, recall, NDCG, MRR and MAP at k = 1, 3, 5, 10. **NDCG@10** is
the number to compare.

Two backends, because the models score differently:

| `--backend` | recipes | how |
|---|---|---|
| `dense` | 1, 3, 4 | cosine similarity between single vectors |
| `late-interaction` | 2 | MaxSim over zero-padded token embeddings, injected through the evaluator's `score_functions` hook |

Two things to get right, both covered in [`shared/README.md`](shared/README.md):

- **Pooling.** A checkpoint with no Sentence Transformers config silently gets
  mean pooling. Qwen3-Embedding needs `--pooling lasttoken --normalize`.
- **Prompts.** Models trained with an instruction prefix must be evaluated with
  the same one via `--query-prompt` / `--corpus-prompt`.

Note that recipe 3's dense-only evaluation understates it: BGE-M3's sparse and
ColBERT heads are not scored by a cosine evaluator.

## Environments

Each folder is an independent `uv` project with its own `pyproject.toml` and
`uv.lock`:

```bash
cd <folder>
uv sync                              # exact locked versions
uv sync --extra late-interaction     # shared/ only: adds PyLate for recipe 2's eval
uv sync --extra faiss                # shared/ only: adds faiss-cpu for --use-faiss mining
uv run python train.py --dry-run
```

They are separate on purpose: FlagEmbedding, PyLate and ms-swift pull
incompatible pins, and one shared environment would mean fighting the resolver
instead of training models. The cost is one PyTorch install per folder.

`bash run.sh` calls `uv run` internally, so a synced folder needs no activation.

## Suggested reading order

1. `shared/README.md` — the data, the mining filters, the metrics.
2. `01_sentence_transformer/README.md` — the loss table. Most of what you learn
   there transfers to the other three.
3. Then whichever of 2, 3, 4 matches the problem you have.

## Relation to the rest of this repository

This handoff is a teaching-shaped extraction. The research code it came from
lives in `../src`, is wired to six tool-retrieval benchmarks rather than a
generic dataset, and carries the PORTS preference loss
([Molfetta et al., EMNLP 2025](https://aclanthology.org/2025.emnlp-main.507/)),
which is deliberately **not** included here — learn the baselines first.
`../CLAUDE.md` is the guide to that code.

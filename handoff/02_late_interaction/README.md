# 02 — Late interaction / ColBERT (PyLate)

A dense bi-encoder squeezes a whole passage into one vector, which throws away
term-level detail. A late-interaction model keeps **one small vector per token**
and defers the comparison to scoring time:

```
MaxSim(q, d) = sum over query tokens t of ( max over document tokens s of  q_t · d_s )
```

Each query token finds its own best match anywhere in the document, and those
matches are summed. The result is markedly better recall on queries that hinge
on a specific rare term, at the cost of an index that is one vector per token
instead of one per document.

## Setup

```bash
cd handoff/02_late_interaction
uv sync
```

Requires `../data/train.jsonl` — build it first with `../shared/prepare_data.sh`.

## Run

```bash
bash run.sh --dry-run
bash run.sh
BATCH_SIZE=64 bash run.sh
```

Then evaluate — note the different backend:

```bash
cd ../shared
uv sync --extra late-interaction     # once, adds PyLate to the eval env
MODEL=../checkpoints/late_interaction/final BACKEND=late-interaction bash evaluate.sh
```

## How the data is used

`train.py` reshapes `../data/train.jsonl` into:

| column | source field |
|---|---|
| `query` | `query` |
| `document` | `positive` |
| `negative_0` … `negative_{N-1}` | first N entries of `negatives` |

As with Sentence Transformers, only column *order* matters. The first column is
tokenized as a query and every column after it as a document — which is exactly
what PyLate's `ColBERTCollator` is for, since queries and documents get
different lengths and (optionally) different prefixes.

Each step therefore encodes `batch_size x (1 + num_negatives)` documents plus
`batch_size` queries. That product, not the batch size alone, is what fills the
GPU.

## Choosing a loss

| `--loss` | class | what it does |
|---|---|---|
| `contrastive` *(default)* | `pylate.losses.Contrastive` | InfoNCE over MaxSim scores: the positive against all explicit negatives plus every other document in the batch. |
| `cached_contrastive` | `pylate.losses.CachedContrastive` | The same objective with GradCache. Encodes in chunks of `--mini-batch-size` so peak memory stops tracking the batch size. |

Both accept `--temperature`, and `train.py` passes it to either one — worth
knowing, because it is easy to write a dispatcher that forwards the temperature
to only one branch and then wonder why swapping the loss changed the results.
(The research code in `../../src/train_colbert.py` has exactly that bug.)

`--temperature` is the sharpness of the contrast:

- **0.05** (default) — suits the mined hard negatives `prepare_data.py` produces.
- **0.2** — suits random or weak negatives, where a sharp temperature just
  amplifies label noise.

## Parameters

### Model

| flag | default | notes |
|---|---|---|
| `--model` | `lightonai/GTE-ModernColBERT-v1` | PyLate's reference modern ColBERT (ModernBERT backbone, 8k context). |
| `--query-length` | 32 | Queries are padded to *exactly* this. NQ questions are short; raise to 64 for verbose queries. |
| `--document-length` | 300 | The main VRAM and index-size knob. Every token becomes a stored vector. |
| `--use-extra-length` | off | Adds 7 positions for mask-token query expansion (original ColBERT recipe). |
| `--use-prompts` | off | Prepends `search_query: ` / `search_document: `. |

Alternatives worth trying: `answerdotai/answerai-colbert-small-v1` (33M, very
fast) and `lightonai/LateOn` / `lightonai/LateOn-Code` (ModernBERT, the
checkpoints the parent research repo uses).

**Only enable `--use-prompts` for a model that was trained with those prefixes.**
ColBERT-Zero expects them; GTE-ModernColBERT and LateOn do not, and adding them
costs accuracy. Passing a non-ColBERT encoder to `--model` also works, but PyLate
then bolts on a randomly initialised projection head — fine as a starting point
for real training, useless if you evaluate it untrained.

### Optimisation

| flag | default | notes |
|---|---|---|
| `--batch-size` | 32 | |
| `--grad-accum` | 8 | **Keep at 8 or above.** To fit VRAM, lower `--batch-size` or `--num-negatives` instead. The recipes are tuned around the effective batch. |
| `--lr` | 3e-6 | Much lower than the dense recipe. These checkpoints are already strong retrievers and a high LR destroys them in a few hundred steps. |
| `--num-negatives` | 4 | Directly multiplies documents encoded per step. |
| `--gradient-checkpointing` | on in `run.sh` | Practically required. `train.py` warns if you leave it off. |
| `--epochs` | 1 | |

`train.py` prints the effective batch and the per-step document count before
training, and warns when `--grad-accum` is below 8 or gradient checkpointing is
off.

### Evaluation during training

`--eval-steps` runs a small MaxSim recall/NDCG evaluator over `--eval-queries`
(default 200) queries against `--eval-corpus` (default 2000) documents. Keep both
small: MaxSim cost grows with *query tokens x document tokens*, so a full
evaluation mid-training is far more expensive than the dense equivalent. Use
`../shared/evaluate.py` for the real numbers.

### Output

`--output-dir` defaults to `../checkpoints/late_interaction/`, with the model in
`final/` and a `training_meta.json` alongside it. The saved directory is a
standard PyLate checkpoint: `pylate.models.ColBERT(model_name_or_path=...)`
loads it back.

## Costs to be aware of

Late interaction is not free. Against the dense recipe:

- **Index size** — one vector per token instead of one per document, typically
  100–300x more storage. Production systems use PLAID compression or token
  pooling (`pool_factor` in PyLate's `encode`) to claw that back.
- **Scoring** — MaxSim is a matrix product per query-document pair, not a dot
  product. Brute force is only viable for small corpora; real deployments
  retrieve candidates with an approximate index first.
- **Evaluation memory** — `../shared/evaluate.py` blocks the MaxSim computation
  over both axes for this reason. If it OOMs, lower `--score-query-chunk` and
  `--score-doc-chunk`.

The shared evaluator zero-pads token embeddings into a rectangular tensor before
scoring. That is safe because MaxSim takes a max over document tokens and padded
positions score 0, which never beats a genuine match between normalized
embeddings.

## Reference

- [PyLate documentation](https://lightonai.github.io/pylate/)
- ColBERT: [Khattab & Zaharia, 2020](https://arxiv.org/abs/2004.12832);
  ColBERTv2: [Santhanam et al., 2021](https://arxiv.org/abs/2112.01488)
- PyLate ships **no** trainer class. Training goes through
  `sentence_transformers.SentenceTransformerTrainer` with
  `pylate.utils.ColBERTCollator` as the data collator — that pairing is the one
  non-obvious part of this script.

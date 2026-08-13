# 01 — Dense bi-encoder (Sentence Transformers)

One encoder, one vector per text, cosine similarity for relevance. This is the
standard retrieval recipe and the baseline the other three should beat. Start
here.

```
query  ──> encoder ──> [768] ─┐
                              ├─ cosine similarity
passage ─> encoder ──> [768] ─┘        (same weights)
```

## Setup

```bash
cd handoff/01_sentence_transformer
uv sync
```

Requires `../data/train.jsonl` — build it first with `../shared/prepare_data.sh`.

## Run

```bash
bash run.sh --dry-run          # resolve config and data, train nothing
bash run.sh                    # defaults: all-MiniLM-L6-v2, MNRL, 1 epoch
LOSS=cmnrl BATCH_SIZE=256 bash run.sh
```

Then evaluate:

```bash
cd ../shared
MODEL=../checkpoints/sentence_transformer/final bash evaluate.sh
```

## How the data is used

`train.py` reads `../data/train.jsonl` and reshapes it into the column layout
Sentence Transformers expects. **Column names are irrelevant — only their order
matters.** The first column is the anchor, the second the positive, and every
column after that is a negative:

| column | source field |
|---|---|
| `anchor` | `query` |
| `positive` | `positive` |
| `negative_1` … `negative_N` | first N entries of `negatives` |

`--num-negatives` decides N. It is capped by what `prepare_data.py` mined and by
what the chosen loss can consume.

Beyond the explicit negatives, MNRL-family losses treat **every other row's
passage in the same batch as an additional negative**. That is why batch size
matters so much here: a batch of 64 with 4 hard negatives gives each query
4 + 63 = 67 negatives to discriminate against. `BatchSamplers.NO_DUPLICATES` is
enabled so a text never lands in the same batch twice, which would make it its
own negative.

## Choosing a loss

`--loss` picks the objective. Each one wants a different column layout, and
`train.py` builds the right one for you:

| `--loss` | class | columns consumed | in-batch negatives | extra flags |
|---|---|---|---|---|
| `mnrl` *(default)* | `MultipleNegativesRankingLoss` | anchor, positive, all negatives | yes | `--scale` |
| `cmnrl` | `CachedMultipleNegativesRankingLoss` | anchor, positive, all negatives | yes | `--mini-batch-size` |
| `mnrl-symmetric` | `MultipleNegativesSymmetricRankingLoss` | anchor, positive **only** | yes | `--scale` |
| `triplet` | `TripletLoss` | anchor, positive, `negative_1` **only** | no | `--triplet-margin` |
| `gist` | `GISTEmbedLoss` | anchor, positive, all negatives | yes | `--guide-model`, `--gist-temperature` |

**`mnrl`** — cross-entropy over `[positive, all negatives]` scaled by `--scale`
(20.0 means a temperature of 0.05). The default for a reason: it is stable, uses
in-batch negatives for free, and improves monotonically with batch size.

**`cmnrl`** — the same objective with GradCache. It runs the encoder in chunks of
`--mini-batch-size`, caches the embeddings, and backpropagates per chunk, so
peak memory depends on the *mini*-batch, not the batch. Use it when you want a
batch of 512+ on one GPU. It is slower per step (two forward passes) but almost
always worth it, because MNRL quality tracks batch size.

**`mnrl-symmetric`** — scores query→passage *and* passage→query. Helps when both
sides are natural queries (duplicate-question retrieval). It ignores explicit
negatives, so mining them is wasted effort with this loss.

**`triplet`** — margin loss on a single negative, no in-batch negatives. Much
weaker for retrieval; included because it is the textbook loss and worth seeing
lose to `mnrl` on the same data.

**`gist`** — MNRL plus a guide model that removes candidates it considers
*more* similar to the anchor than the positive is, i.e. it filters false
negatives at training time rather than mining time. Useful when your negatives
are noisy. Costs a second encoder forward pass per step.

### Losses this data cannot feed

`prepare_data.py` emits triplets, not labels, so these need a different dataset:

| loss | needs |
|---|---|
| `CoSENTLoss`, `CosineSimilarityLoss`, `AnglELoss` | `sentence1, sentence2, score` — a float similarity per pair |
| `MarginMSELoss` | `query, positive, negative, label` — a teacher score *margin* |
| `ContrastiveLoss`, `OnlineContrastiveLoss` | `sentence1, sentence2, label` — binary 0/1 |

`prepare_data.py --with-scores` gets you part way: it adds `positive_score` and
`negative_scores` from the mining model, which is the teacher signal
`MarginMSELoss` wants.

## Parameters

### Model

| flag | default | notes |
|---|---|---|
| `--model` | `sentence-transformers/all-MiniLM-L6-v2` | 22M params, fast enough to iterate on a laptop GPU. Swap to `BAAI/bge-base-en-v1.5` (109M) or `Alibaba-NLP/gte-base-en-v1.5` for real numbers. |
| `--max-seq-length` | 384 | NQ passages are long; below ~256 you start truncating away the answer. |
| `--pooling` | model's own | Only pass this for a raw transformers checkpoint with no Sentence Transformers config. Adds Pooling + Normalize. |

### Prompts

Some models expect an instruction prefix and lose accuracy without it. BGE v1.5
is the common case:

```bash
bash run.sh --model BAAI/bge-base-en-v1.5 \
    --query-prompt "Represent this sentence for searching relevant passages: "
```

Whatever you train with, pass the same prompts to `../shared/evaluate.py`.

### Optimisation

| flag | default | notes |
|---|---|---|
| `--batch-size` | 64 | The single most important knob for MNRL quality. Raise it until VRAM complains, then switch to `--loss cmnrl`. |
| `--grad-accum` | 1 | Increases the *optimizer* batch but **not** the in-batch-negative pool — accumulation does not help MNRL the way a bigger real batch does. Prefer `cmnrl`. |
| `--lr` | 2e-5 | 2e-5 for a from-scratch-ish encoder; drop to 1e-5 or 5e-6 when fine-tuning an already-strong retriever, which is easy to wreck. |
| `--epochs` | 1 | One pass is usually enough with 50k pairs. More overfits fast. |
| `--warmup-ratio` | 0.1 | |
| `--scheduler` | `cosine` | |
| `--gradient-checkpointing` | off | Trades ~30% speed for a large VRAM saving. Not needed at these model sizes. |
| `--max-steps` | -1 | Set to a small number for smoke runs. |

### Evaluation during training

`--eval-steps` (default 200) runs an `InformationRetrievalEvaluator` over
`--eval-queries` (default 500) held-out queries against a capped corpus. It is a
progress signal, not the final number — use `../shared/evaluate.py` on the full
eval set for that. `--no-eval` turns it off.

### Output

Checkpoints land in `--output-dir` (default
`../checkpoints/sentence_transformer/`), with the model to evaluate in `final/`
and a `training_meta.json` recording what produced it. `--no-save` trains
without writing anything.

## Suggested experiments

1. `bash run.sh` — the baseline.
2. `NUM_NEGATIVES=0 bash run.sh` — in-batch negatives only. Shows what the mining
   step actually bought you.
3. `LOSS=cmnrl BATCH_SIZE=512 bash run.sh` — the batch-size effect.
4. `LOSS=triplet bash run.sh` — why the field moved away from margin losses.

Evaluate all four with `../shared/evaluate.sh` and put NDCG@10 side by side.

## Reference

- [Sentence Transformers training overview](https://sbert.net/docs/sentence_transformer/training_overview.html)
- [Loss overview table](https://sbert.net/docs/sentence_transformer/loss_overview.html) —
  the authoritative map from data format to loss.
- `MultipleNegativesRankingLoss` in Sentence Transformers 5.x also accepts
  `directions` and `hardness_mode` for symmetric and hardness-weighted variants;
  `train.py` does not expose them, but they are a one-line change in
  `build_loss()`.

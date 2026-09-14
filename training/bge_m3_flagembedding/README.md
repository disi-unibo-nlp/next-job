# 03 — BGE-M3 (FlagEmbedding)

BGE-M3 is one backbone (XLM-RoBERTa-large, 568M, 100+ languages, 8192 tokens)
with **three retrieval heads trained simultaneously**:

| head | output | scoring | closest analogue |
|---|---|---|---|
| dense | one CLS vector | cosine | recipe 01 |
| sparse | a learned weight per vocabulary term | lexical overlap | BM25 |
| colbert | one vector per token | MaxSim | recipe 02 |

That is the whole point of this recipe. Recipes 01 and 02 make you pick between
semantic and term-level matching; BGE-M3 trains both and lets you combine their
scores at retrieval time. `--use-self-distill` goes further: the ensemble of the
three heads becomes a teacher for each individual head, so the dense head learns
from the sparse head's lexical precision and vice versa.

Training does **not** go through Sentence Transformers. The sparse/ColBERT heads
and the self-distillation loop live inside FlagEmbedding's
`EncoderOnlyEmbedderM3Runner`, so `train.py` converts the data and drives that
runner directly.

## Setup

```bash
cd handoff/03_bge_m3_flagembedding
uv sync
```

Requires `../data/train.jsonl` — build it first with `../shared/prepare_data.sh`.

## Run

```bash
bash run.sh --dry-run          # converts the data, prints the plan, trains nothing
bash run.sh
NEG_NUM=15 BATCH_SIZE=1 bash run.sh --gradient-checkpointing
```

Then evaluate the dense head (the shared evaluator is cosine-only, see
[Evaluating](#evaluating) below):

```bash
cd ../shared
MODEL=../checkpoints/bge_m3 bash evaluate.sh \
    --pooling cls --normalize \
    --query-prompt "Represent this sentence for searching relevant passages: "
```

## How the data is used

FlagEmbedding reads its own JSONL format. `train.py` writes it to
`../checkpoints/bge_m3/ft_data/training.jsonl`, one object per query:

```json
{"query": "when does next episode of the ranch come out",
 "pos": ["The Ranch (TV series) All episodes are named after ..."],
 "neg": ["...", "...", "..."],
 "prompt": "",
 "type": "normal"}
```

| field | meaning |
|---|---|
| `query` | the query string |
| `pos` | list of positives. This pipeline always writes exactly one. |
| `neg` | list of negatives, `--neg-num` of them |
| `prompt` | per-example instruction prefix. Empty here; the global one comes from `--query-instruction`. |
| `type` | dataset tag, used by `same_dataset_within_batch`. `"normal"` for everything. |

With `--knowledge-distillation` two more fields appear, carrying the mining
model's similarity scores as a teacher signal:

```json
{"...": "...", "pos_scores": [0.82], "neg_scores": [0.61, 0.58, 0.55]}
```

Those only exist if the shared data was built with `--with-scores`. `train.py`
fails with an explicit message if you ask for KD without them, rather than
training on a silently degraded objective:

```bash
cd ../shared && uv run python prepare_data.py --with-scores --overwrite
```

### `--train-group-size` must equal `--neg-num + 1`

`train_group_size` is how many passages FlagEmbedding puts in each query's
group: one positive plus its negatives. If it disagrees with the number of
negatives actually in the file, the loss quietly trains on padded or truncated
groups.

`train.py` derives it from `--neg-num` automatically. `--train-group-size` exists
as an override and warns when the two disagree. (The research code in
`../../src` keeps them in sync by hand in the YAML — a trap worth knowing about
if you go back to it.)

## Parameters

### Heads and loss

| flag | default | notes |
|---|---|---|
| `--unified-finetuning` | on | Train dense + sparse + colbert. `--dense-only` turns it off and makes this roughly recipe 01 with a bigger model. |
| `--use-self-distill` | on | The ensemble teaches each head. `--no-self-distill` disables it. |
| `--self-distill-start-step` | 0 | Delay self-distillation until the heads are less random. Nonzero is worth trying on long runs. |
| `--temperature` | 0.02 | Much sharper than the dense recipe's 0.05. BGE-M3's published recipe uses 0.02. |
| `--sentence-pooling-method` | `cls` | BGE-M3 pools at CLS. Do not change unless you know the checkpoint differs. |
| `--knowledge-distillation` | off | See above. Needs `pos_scores` / `neg_scores`. |
| `--kd-loss-type` | `m3_kd_loss` | `m3_kd_loss` or `kl_div`. Only used with KD on. |
| `--negatives-cross-device` | on | Shares negatives across GPUs. On one GPU it is a no-op, but it is why `train.py` initialises a single-process distributed group. |

### Lengths

| flag | default | notes |
|---|---|---|
| `--query-max-len` | 128 | NQ questions are short. |
| `--passage-max-len` | 512 | The dominant memory term. BGE-M3 supports 8192, but a long window on 500k passages is very expensive. |
| `--query-instruction` | `"Represent this sentence for searching relevant passages: "` | BGE's retrieval prefix. Pass `''` to train without it — but then evaluate without it too. |

### Optimisation

| flag | default | notes |
|---|---|---|
| `--batch-size` | 2 | Per device. Real cost is `batch-size x (neg-num + 1)` passages of `passage-max-len` tokens through a 568M model. |
| `--neg-num` | 7 | |
| `--grad-accum` | 8 | |
| `--lr` | 1e-5 | |
| `--gradient-checkpointing` | off | Roughly halves memory for ~30% slower steps. Turn it on before lowering `--neg-num`. |
| `--deepspeed` | `ds_stage0.json` | Shipped alongside `train.py`. Stage 0 means no sharding, just DeepSpeed's optimizer and scheduler. Pass `--deepspeed ''` to skip DeepSpeed entirely; move to a stage-2 config for multi-GPU. |

### Multi-GPU

`run.sh` launches a single process. For several GPUs, use `torchrun` so
`negatives_cross_device` has something to share:

```bash
uv run torchrun --nproc_per_node 4 train.py --batch-size 2 --grad-accum 2
```

Note that `train.py` initialises its own single-process group when launched
plainly; under `torchrun` the environment variables are already set and that
branch is skipped.

## Evaluating

The trained model is a transformers checkpoint. `../shared/evaluate.py` scores
the **dense head only** — it computes cosine similarity, which is what the dense
vector is for. Sparse and ColBERT scoring needs FlagEmbedding's own
`BGEM3FlagModel`:

```python
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel("../checkpoints/bge_m3", use_fp16=True)
out = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=True)
```

So a dense-only evaluation *understates* what unified fine-tuning bought you.
Keep that in mind when putting this recipe's NDCG@10 next to recipe 01's.

**`--pooling cls --normalize` is required.** FlagEmbedding writes a plain
transformers directory — no `modules.json`, no `1_Pooling/` — so
`SentenceTransformer(path)` silently falls back to mean pooling, while BGE-M3
pools at CLS. On a 358-document smoke corpus that mistake alone cost ~4 NDCG@10
points (93.3 vs 97.2); on a realistic corpus the gap is much wider, and it looks
like a bad training run rather than a configuration error.

Always pass the same `--query-prompt` you trained with, for the same reason.

## Reference

- [FlagEmbedding repository](https://github.com/FlagOpen/FlagEmbedding)
- BGE-M3 paper: [Chen et al., 2024](https://arxiv.org/abs/2402.03216)
- The argument classes this script fills in are
  `EncoderOnlyEmbedderM3{Model,Data,Training}Arguments` from
  `FlagEmbedding.finetune.embedder.encoder_only.m3`; read them when you need a
  knob `train.py` does not expose.

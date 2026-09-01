# JobHop Wiki Experiment

This repository includes a JobHop-specific adaptation of the LLM-Wiki idea.
The goal is not to reproduce the original Wiki QA pipeline directly. JobHop is
last-step next-occupation prediction, so the main signal is career-transition
structure rather than document evidence routing.

## Data Protocol

The experiment follows the existing JobHop v2 protocol used by the dense and
late-interaction evaluations:

- training and evaluation data are exported under `handoff/data_full`
- the candidate pool is the train-only ESCO occupation pool
- validation and test queries are filtered to targets in the train candidate pool
- metrics are reported with the JobHop evaluator: Accuracy, MRR, NDCG, Recall,
  and MAP at the configured K values

## Frozen Entity Wiki

`scripts/jobhop_qwen_make_entity_pages.py` builds a frozen entity Wiki from the
2,982 ESCO occupations in `handoff/data_full/eval_corpus.jsonl`.

The synced result summary is stored under:

```text
results/wiki_eval/qwen_entity_wiki_full/
```

The frozen Wiki generation completed with 2,982 entity pages and 0 failures.

## Query-Time Pipeline

The latest query-time Wiki experiment uses:

```text
validation/test query
  -> train transition KG
  -> Wiki BM25/grep fallback
  -> raw occupation fallback
  -> Top-100 candidate occupation IDs
  -> optional Qwen direct selection of Top-3 IDs
  -> JobHop metrics
```

The main script is:

```text
scripts/jobhop_qwen_wiki_direct_select.py
```

This script intentionally disables query-time Qwen routing:

```text
ROUTING_DISABLED_FOR_DIRECT_SELECT
```

The reason is empirical. In this JobHop adaptation, Qwen routing over Wiki pages
had very low gold coverage (`gold_in_selected_pages_rate` around 0.03), while a
transition KG built from `handoff/data_full/train.jsonl` raised candidate recall
to around 0.68 to 0.70. This suggests that query-time Wiki page routing is not a
good candidate generator for JobHop, where the target is a plausible next job
rather than a semantically matching document.

## Direct Selection Variant

The current direct-selection variant follows the suggested Option 2:

```text
Top-100 candidates -> one Qwen call -> selected_ids Top-3
```

Qwen is asked to directly select occupation IDs from the candidate list. It does
not score every candidate. The selected IDs are moved to the front, and the
remaining candidates keep the original retrieval order.

Current small-scale findings:

- n=200, c24, Top-3, 128 output tokens: 0 failures, about 0.76 seconds per case
- candidate_recall@100 is 0.705
- retrieval-only MRR@10 is 0.1057
- Qwen direct-select MRR@10 is 0.0946
- n=500 direct select also underperforms retrieval-only

Therefore, the direct-selection Qwen step is useful as an ablation, but the
current evidence does not support using it as the main full-data method.

## Recommended Main Result

The strongest current Wiki-style mainline is:

```text
frozen Qwen entity Wiki + train transition KG + deterministic Top-100 retrieval
```

Qwen routing and Qwen direct selection should be reported as ablations unless
future tuning shows a stable improvement over retrieval-only ranking.

## Synced Results

Formal dense and late-interaction results are organized under:

```text
results/formal_eval/
```

Wiki experiment summaries are organized under:

```text
results/wiki_eval/
```

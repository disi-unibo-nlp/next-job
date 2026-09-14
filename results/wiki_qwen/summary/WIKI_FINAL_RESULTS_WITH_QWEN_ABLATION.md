# Wiki Final Results with Qwen Ablation

This package contains the final Wiki/KG full-scale results and query-time Qwen ablation results for JobHop.

## Main Full-Scale Experiments

The main experiments are CPU-only and do not use query-time Qwen, vLLM, neural retrievers, or per-candidate LLM scoring.

The transition KG is built only from `handoff/data_full/train.jsonl`. Validation/test trajectories are used only for evaluation.

| Split | Method | N | Acc@1 | MRR@10 | NDCG@10 | Recall@10 | MAP@10 | Candidate Recall@100 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| validation | wiki_only | 32123 | 0.030445 | 0.058138 | 0.076403 | 0.136133 | 0.058138 | 0.317592 |
| validation | kg_alone | 32123 | 0.069483 | 0.133287 | 0.176132 | 0.316751 | 0.133287 | 0.675528 |
| validation | wiki_kg_fallback | 32123 | 0.069483 | 0.133287 | 0.176132 | 0.316751 | 0.133287 | 0.675528 |
| validation | wiki_kg_score_fusion_alpha_0.8 | 32123 | 0.072378 | 0.138645 | 0.182433 | 0.325717 | 0.138645 | 0.625782 |
| test | wiki_only | 32089 | 0.030073 | 0.056037 | 0.073193 | 0.129203 | 0.056037 | 0.308860 |
| test | kg_alone | 32089 | 0.069744 | 0.133072 | 0.176111 | 0.317430 | 0.133072 | 0.671040 |
| test | wiki_kg_fallback | 32089 | 0.069744 | 0.133072 | 0.176111 | 0.317430 | 0.133072 | 0.671040 |
| test | wiki_kg_score_fusion_alpha_0.8 | 32089 | 0.072579 | 0.137745 | 0.181313 | 0.323943 | 0.137745 | 0.623796 |

Main observation:

```text
Wiki-only << KG-alone ~= Wiki+KG fallback < Wiki+KG score fusion
```

This indicates that JobHop is mainly driven by train-derived occupation transition statistics. Frozen Wiki text alone is weak, but score-level Wiki+KG fusion gives a small and consistent improvement over KG-alone.

## Query-Time Qwen Ablations

Qwen ablations are run only on 2,000 validation instances. They are used to decide whether query-time Qwen should be included in the full-scale pipeline.

| Ablation | N | Route Failures | Ranking Failures | Gold in Selected Pages | Baseline MRR@10 | Qwen MRR@10 | Delta MRR@10 | Baseline NDCG@10 | Qwen NDCG@10 | Delta NDCG@10 | Baseline Recall@10 | Qwen Recall@10 | Delta Recall@10 | Seconds/Case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen_direct_select_top3_clean | 2000 | 0 | 0 | 0.000000 | 0.133909 | 0.102407 | -0.031502 | 0.175972 | 0.149313 | -0.026659 | 0.314000 | 0.305000 | -0.009000 | 0.683751 |
| qwen_routing_clean | 2000 | 0 | 265 | 0.039500 | 0.133909 | 0.134373 | 0.000464 | 0.175972 | 0.175087 | -0.000886 | 0.314000 | 0.308000 | -0.006000 | 1.170198 |

Qwen ablation observation:

- Direct selection is stable after lowering concurrency, but it underperforms the retrieval-only baseline.
- Routing has very low gold-page coverage and gives negligible or negative gains while still introducing ranking failures.
- Therefore, query-time Qwen is kept as an ablation and is not used in the full-scale main Wiki pipeline.

## Included Files

- `scripts/jobhop_wiki_kg_ablation.py`
- `scripts/jobhop_wiki_kg_score_fusion.py`
- `scripts/create_named_wiki_pages.py`
- `scripts/jobhop_qwen_wiki_direct_select.py`
- `scripts/jobhop_qwen_wiki_query_eval.py`
- `outputs/wiki_final_results_with_qwen_ablation_summary/`
- full-scale main metrics and run configs
- Qwen ablation metrics and run configs
- frozen Wiki metadata and named Wiki pages

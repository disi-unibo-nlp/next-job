# Which Results To Use

This file records the final result policy for the cleaned project.

## Use these as final

### Formal dense and transition results

Use:

- `results/formal_eval/formal_eval_summary.csv`
- `results/paper_checks/bge_m3_original_formal_per_query_full_test/`
- `results/paper_checks/trained_st_formal_per_query_full_test/`
- `results/paper_checks/significance_tests_fixed/`

Key test values:

| System | MRR@10 | NDCG@10 | Recall@10 |
| --- | ---: | ---: | ---: |
| BGE-M3 pure | 0.0685488 | 0.0890462 | 0.1554115 |
| BGE-M3 + Markov | 0.1396752 | 0.1790239 | 0.3071458 |
| Trained ST pure | 0.1108547 | 0.1492578 | 0.2750164 |
| Trained ST + Markov | 0.1547090 | 0.1982009 | 0.3398361 |
| Trained Qwen3 Embedding + Markov | 0.1516157 | 0.1947235 | 0.3355667 |
| LateOn + Markov | 0.1331884 | 0.1719540 | 0.2985135 |

### Wiki/KG and Qwen results

Use:

- `results/wiki_qwen/wiki_kg_ablation_test_full/test/metrics.json`
- `results/wiki_qwen/wiki_kg_score_fusion_test_alpha08/test/metrics.json`
- `results/wiki_qwen/summary/wiki_main_full_results.csv`
- `results/wiki_qwen/summary/qwen_ablation_validation_n2000.csv`

Final Wiki/KG interpretation:

- KG-alone is strong because it uses train-derived transition structure.
- Wiki-only is weaker as a standalone ranker.
- Wiki+KG score fusion improves over KG-alone but does not exceed the best trained dense + Markov hybrid.
- Query-time Qwen ablations are validation-only and should be described as ablations, not full-test main results.

### Teacher-requested diagnostic outputs

Use:

- `results/paper_checks/error_analysis_plus_bge/`
- `results/paper_checks/filter_counts/`
- `results/paper_checks/provenance_checks_fast/`
- `results/paper_checks/worked_example_search/`
- `docs/paper_checks_provenance_answer_summary.md`

## Do not use as final

Earlier BGE `data_full` significance outputs were useful while debugging, but the formal BGE export supersedes them. The final manuscript and significance claims should use `bge_m3_original_formal_per_query_full_test`.

Old sync tarballs and intermediate bundles are not part of this clean project. They can be archived outside the repository if needed, but should not be presented as final outputs.

## Short answer for the paper

If the paper mentions BGE-M3, use the formal numbers:

- BGE-M3 pure MRR@10: 0.0685
- BGE-M3 + Markov MRR@10: 0.1397
- Paired bootstrap confirms the fusion gain is significant.

If the paper compares the best trained hybrid against BGE-M3 fusion:

- Trained ST + Markov MRR@10: 0.1547
- BGE-M3 + Markov MRR@10: 0.1397
- Paired bootstrap confirms the trained ST hybrid gain is significant.


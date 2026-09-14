# Error Analysis Plus BGE

This table extends the existing KG/Wiki error analysis with formal BGE-M3 and trained ST per-query ranks.

## Inputs
- error_per_case: `outputs/paper_checks/error_analysis/error_analysis_per_case.csv`
- bge_ranks: `outputs/paper_checks/bge_m3_original_formal_per_query_full_test/bge_m3_formal_per_query_ranks.csv`
- trained_st_ranks: `outputs/paper_checks/trained_st_formal_per_query_full_test/trained_st_formal_per_query_ranks.csv`

## Overall

| method                         |     n |   Accuracy@1 |    MRR@10 |   NDCG@10 |   Recall@10 |
|:-------------------------------|------:|-------------:|----------:|----------:|------------:|
| KG-alone                       | 32089 |    0.0697435 | 0.133072  | 0.176111  |    0.31743  |
| wiki_kg_score_fusion_alpha_0.8 | 32089 |    0.0725794 | 0.137745  | 0.181313  |    0.323943 |
| BGE-M3 pure                    | 32089 |    0.0343731 | 0.0685488 | 0.0890462 |    0.155412 |
| BGE-M3+Markov fusion           | 32089 |    0.077659  | 0.139675  | 0.179024  |    0.307146 |
| trained ST pure                | 32089 |    0.0526037 | 0.110855  | 0.149258  |    0.275016 |
| trained ST+Markov fusion       | 32089 |    0.0863847 | 0.154709  | 0.198201  |    0.339836 |

## Files Written
- `error_analysis_plus_bge_per_case_long.csv`
- `error_analysis_plus_bge_by_bucket.csv`
- `error_analysis_plus_bge_examples.jsonl`
- `error_analysis_plus_bge_summary.md`

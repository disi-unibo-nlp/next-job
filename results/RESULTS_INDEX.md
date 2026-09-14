# Results Index

This directory contains curated final outputs for the paper and teacher-requested checks.

## `formal_eval/`

Compact summary of the main formal evaluation results.

Important file:

- `formal_eval_summary.csv`

Use this for the main table covering dense retrievers and transition-fused systems.

## `paper_checks/`

Final diagnostic and validation outputs added after review.

Important subdirectories:

- `bge_m3_original_formal_per_query_full_test/`: formal BGE-M3 pure and BGE-M3 + Markov per-query ranks, metrics, selected alpha, and runtime metadata.
- `trained_st_formal_per_query_full_test/`: trained Sentence Transformer pure/fusion metrics and paired significance against BGE-M3 fusion.
- `significance_tests_fixed/`: paired bootstrap significance tests for fixed final comparisons.
- `error_analysis_plus_bge/`: error analysis including KG, Wiki+KG, BGE-M3, and trained ST systems.
- `filter_counts/`: exact split-level filtering counts.
- `lateon_only_datafull_full_test/`: LateOn-only check.
- `provenance_checks_fast/`: code provenance snippets for KG parameters, Wiki alpha, Qwen failure handling, and candidate recall.
- `worked_example_search/`: candidate search and selected worked-example outputs.

## `wiki_qwen/`

Wiki/KG and query-time Qwen outputs.

Important subdirectories:

- `wiki_kg_ablation_test_full/`: test-set KG/Wiki ablation.
- `wiki_kg_score_fusion_test_alpha08/`: test-set Wiki+KG score fusion.
- `qwen_direct_select_validation_n2000/`: validation-only Qwen direct selection ablation.
- `qwen_routing_validation_n2000/`: validation-only Qwen routing/reranking ablation.
- `summary/`: paper-facing summaries.

## Result precedence

When two files appear to answer the same question, use this order:

1. `results/paper_checks/` for teacher-requested formal checks.
2. `results/formal_eval/` for compact main dense/hybrid tables.
3. `results/wiki_qwen/summary/` for Wiki/KG and Qwen tables.
4. Older tarballs or exploratory outputs outside this clean project only for archival debugging.


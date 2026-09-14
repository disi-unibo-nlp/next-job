# Source Scripts

This directory contains the experiment and analysis scripts used by the project.

## Main evaluation scripts

- `dense_eval.py`: dense retrieval evaluation for semantic encoders.
- `lateon_eval.py`: LateOn/late-interaction evaluation.
- `jobhop_eval_common.py`: shared evaluation utilities.
- `jobhop_transition_wiki_eval.py`: transition/Wiki evaluation support.
- `jobhop_wiki_kg_ablation.py`: KG and Wiki ablation.
- `jobhop_wiki_kg_score_fusion.py`: Wiki+KG score fusion.

## Final paper-check scripts

- `export_bge_m3_formal_per_query.py`: formal BGE-M3 per-query export and BGE+Markov metrics.
- `export_trained_st_formal_per_query_and_sig.py`: trained ST formal export and paired significance against BGE-M3 fusion.
- `error_analysis_plus_bge.py`: extended error analysis including BGE-M3.
- `filter_counts_formal.py`: exact filter counts by split.
- `eval_lateon_only_datafull.py`: LateOn-only check.
- `search_worked_example_wiki_kg_fast.py`: search for worked-example candidates.
- `export_worked_example_single_case.py`: export top candidates for one worked-example case.

## Qwen/Wiki scripts

- `jobhop_qwen_make_entity_pages.py`: generate occupation Wiki pages.
- `jobhop_qwen_wiki_direct_select.py`: Qwen direct-selection validation ablation.
- `jobhop_qwen_wiki_query_eval.py`: Qwen routing/reranking validation ablation.
- `jobhop_llm_wiki_rerank.py`: LLM Wiki reranking support.

Most scripts expect the full exported JobHop data and/or model checkpoints, which are not committed in this clean repository.


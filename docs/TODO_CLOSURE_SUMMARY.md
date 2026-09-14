# Teacher TODO Closure Summary

This file summarizes the experimental work completed to address the teacher's main TODOs.

## 1. Clarify BGE formal evaluation and significance

Completed.

Evidence:

- `src/export_bge_m3_formal_per_query.py`
- `results/paper_checks/bge_m3_original_formal_per_query_full_test/`

Outcome:

- BGE-M3 pure MRR@10: 0.0685488
- BGE-M3 + Markov MRR@10: 0.1396752
- Paired bootstrap over 32,089 test cases shows significant gains for fusion.

This replaces the older diagnostic `data_full` BGE significance output.

## 2. Add trained ST fusion vs BGE-M3 paired significance

Completed.

Evidence:

- `src/export_trained_st_formal_per_query_and_sig.py`
- `results/paper_checks/trained_st_formal_per_query_full_test/`

Outcome:

- BGE-M3 + Markov MRR@10: 0.1396752
- Trained ST + Markov MRR@10: 0.1547090
- Delta MRR@10: 0.015034
- Bootstrap 95 percent CI: [0.013619, 0.016457]
- Bootstrap p-value: 0

## 3. Add BGE-M3 to error analysis

Completed.

Evidence:

- `src/error_analysis_plus_bge.py`
- `results/paper_checks/error_analysis_plus_bge/`

The extended table includes:

- KG-alone
- Wiki+KG score fusion
- BGE-M3 pure
- BGE-M3 + Markov
- Trained ST pure
- Trained ST + Markov

This lets the paper discuss whether errors come from semantic retrieval, transition evidence, or Wiki/KG fusion.

## 4. Add per-filter exact counts

Completed.

Evidence:

- `src/filter_counts_formal.py`
- `results/paper_checks/filter_counts/filter_counts_summary.md`
- `results/paper_checks/filter_counts/filter_counts_by_split.csv`

Final counts:

- Train retained trajectories: 256,593
- Validation evaluation cases: 32,123
- Test evaluation cases: 32,089
- Train candidate vocabulary size: 2,982

The output also records unknown-code removal, duplicate collapse, length filtering, and target-in-train-vocabulary filtering by split.

## 5. Clarify KG and Wiki+KG provenance

Completed.

Evidence:

- `results/paper_checks/provenance_checks_fast/provenance_summary.md`
- `docs/paper_checks_provenance_answer_summary.md`

Final interpretation:

- KG parameters were fixed heuristically/development-side before test evaluation.
- Wiki+KG alpha 0.8 comes from the Wiki/KG configuration/validation search and is frozen for test reporting.
- Qwen failures remain in the denominator and use deterministic fallback behavior.
- Qwen candidate recall at 100 stays constant because the Qwen ablations rerank the same top-100 pool.

## 6. Worked example with top candidates

Completed as a result artifact. The manuscript can use the selected example.

Evidence:

- `src/search_worked_example_wiki_kg_fast.py`
- `src/export_worked_example_single_case.py`
- `results/paper_checks/worked_example_search/`

Recommended case:

- Case index: 3521
- History: `administrative assistant -> advertising assistant`
- Gold: `management assistant`

Top-candidate summary:

| Method | Top-1 | Top-2 | Top-3 | Gold rank |
| --- | --- | --- | --- | ---: |
| Transition KG | administrative assistant | sales support assistant | public relations officer | 11 |
| Wiki only | advertising assistant | administrative assistant | legal administrative assistant | 4 |
| Wiki+KG fusion | administrative assistant | management assistant | sales support assistant | 2 |

This case directly supports the paper's claim that frozen Wiki evidence can complement train-derived transition structure.

## Remaining manuscript-only work

The experimental side is covered. Remaining work is mainly writing and presentation:

- Integrate final tables consistently.
- Add or polish Figure 1.
- Insert the worked-example table and text.
- Make sure the narrative states that some Qwen results are validation ablations rather than full-test main results.


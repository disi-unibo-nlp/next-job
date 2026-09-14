# Project Manifest

This is the cleaned submission directory for the JobHop next-occupation prediction project.

## Core entry points

- `README.md`: high-level project explanation and main final numbers.
- `docs/WHICH_RESULTS_TO_USE.md`: final result precedence and values.
- `docs/TODO_CLOSURE_SUMMARY.md`: how the teacher TODOs were addressed.
- `results/RESULTS_INDEX.md`: index of curated result folders.
- `paper/What_Drives_Next_Occupation_Prediction.pdf`: compiled manuscript.

## Code

- `src/`: experiment and analysis code.
- `training/`: training handoff code for task-adapted retrievers.
- `environment/`: Docker/build/Slurm helpers.
- `scripts/smoke_test.sh`: lightweight repository integrity check.
- `scripts/reproduce_main.sh`: reproduction guidance and final-output pointers.

## Final outputs

- `results/formal_eval/formal_eval_summary.csv`: compact formal evaluation summary.
- `results/paper_checks/bge_m3_original_formal_per_query_full_test/`: final formal BGE-M3 results.
- `results/paper_checks/trained_st_formal_per_query_full_test/`: final formal trained ST results and significance.
- `results/paper_checks/significance_tests_fixed/`: paired significance outputs.
- `results/paper_checks/error_analysis_plus_bge/`: extended error analysis.
- `results/paper_checks/filter_counts/`: exact filter counts.
- `results/paper_checks/provenance_checks_fast/`: code provenance checks.
- `results/paper_checks/worked_example_search/`: worked-example candidate search.
- `results/wiki_qwen/`: Wiki/KG and Qwen ablation results.

## Data and model policy

The full dataset export and trained checkpoints are not committed here. The repository includes:

- Metadata for the full data export in `data/full_metadata/`.
- Code required to regenerate or rerun experiments once data/checkpoints are restored.
- Curated final outputs used by the manuscript.

This keeps the GitHub-ready project compact while preserving the evidence needed for review.


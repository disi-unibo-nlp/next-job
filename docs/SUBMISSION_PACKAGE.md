# Submission Package Notes

The GitHub repository keeps the curated final outputs, including per-query evidence files. A course submission ZIP can be smaller because the teacher mainly needs the paper, code, documentation, configs, summaries, and main result tables.

For the ZIP archive, exclude:

- `.git/`
- `__pycache__/`
- local virtual environments and caches
- redundant large per-query JSONL or long-form diagnostic files when the corresponding CSV summaries are included

The helper script below creates a compact archive without deleting anything from the repository:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/make_submission_zip.ps1
```

By default it writes:

```text
next_occupation_prediction_project_submission.zip
```

The archive intentionally excludes:

- `results/paper_checks/lateon_only_datafull_full_test/lateon_only_datafull_per_query.jsonl`
- `results/paper_checks/bge_m3_original_formal_per_query_full_test/bge_m3_formal_per_query_ranks.jsonl`
- `results/paper_checks/trained_st_formal_per_query_full_test/trained_st_formal_per_query_ranks.jsonl`
- `results/paper_checks/error_analysis_plus_bge/error_analysis_plus_bge_per_case_long.csv`

The corresponding summaries, configs, CSV tables, and final metrics remain included.


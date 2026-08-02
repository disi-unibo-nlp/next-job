# Notebook Conversion Changes

Converted:

- `09_jobhop_v2_dense_new_models_eval_colab.ipynb` -> `scripts/dense_eval.py`
- `10_jobhop_v2_lighton_lateon_retrieval_colab.ipynb` -> `scripts/lateon_eval.py`

Protocol preserved:

- JobHop v2 train/validation/test split discovery from Hugging Face parquet files.
- Schema standardization from `resume_id` to `person_id` and `university_level` to `university_studies` when needed.
- Removal of `unknown` occupation codes.
- Stable trajectory ordering by parsed `start_date`, then `end_date`, then `matched_code`.
- Consecutive duplicate occupation collapse.
- Minimum trajectory length of two occupations.
- Last-step next-occupation prediction.
- Train-only candidate occupation pool.
- Validation/test filtering to cases whose target appears in the train candidate pool.
- ESCO metadata loading from `danieldux/ESCO`.
- Candidate and query text templates from the notebooks.
- Markov transition-count baseline.
- Dense embedding retrieval with row min-max embedding normalization and row max-normalized Markov scores.
- LateOn retrieval with PyLate ColBERT, PLAID indexing, per-case LateOn/Markov candidate union, and per-row min-max hybrid scoring.
- MRR, Recall@5, and Recall@10 metrics.

Intentional changes:

- Removed Colab-only `!pip` setup cells. Dependencies now live in `requirements.txt`.
- Removed notebook-only `display()` and Styler rendering. Results are written to CSV and JSON.
- Added non-interactive argparse interfaces for model name, max evaluation cases, random seed, batch size, retrieve-k, output directory, and Hugging Face cache directory.
- Added support for `--max-eval-cases none` for full validation/test evaluation.
- Changed execution from multi-model notebook loops to one model per script invocation and one model per SLURM job.
- Added shared Hugging Face cache environment variables for sbatch execution.
- Fixed the alpha-selection protocol: alpha is selected using validation MRR only, then frozen for one final test evaluation.
- Test metrics are no longer used to select or compare alphas.
- Equal validation MRR ties use the first alpha in the original notebook sweep order.
- Added runtime, GPU information, package versions, dataset statistics, preprocessing configuration, data-file discovery, validation sweeps, selected alpha, final test results, and failure files.
- Added exception handling so partial results are saved when a model fails.
- Added multi-K IR metrics: Accuracy@K, Precision@K, Recall@K, MRR@K, NDCG@K, and MAP@K.
- Added `sentence_transformers.evaluation.InformationRetrievalEvaluator` for pure dense retrieval evaluation.
- Replaced host-level Conda execution with Docker-based SLURM execution following the cluster web guide.

Script-specific notes:

- `scripts/dense_eval.py` keeps the dense alpha grid `[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]`.
- `scripts/dense_eval.py` preserves full candidate scoring; `--retrieve-k` controls ranking extraction depth and is clamped to at least 10 for the reported metrics.
- `scripts/dense_eval.py` writes SentenceTransformers IR evaluator outputs to `sentence_transformers_ir_validation.*` and `sentence_transformers_ir_test.*`.
- `scripts/lateon_eval.py` keeps the LateOn alpha grid `[0.0, 0.25, 0.5, 0.75, 0.9, 1.0]`.
- `scripts/lateon_eval.py` uses `--retrieve-k` for both LateOn retrieval depth and Markov candidate depth, matching the notebook default of 200.

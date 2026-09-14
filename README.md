# What Drives Next-Occupation Prediction?

This repository contains the code, curated outputs, and manuscript artifacts for a JobHop v2 next-occupation prediction study. The task is last-step occupation prediction over a fixed ESCO occupation vocabulary. Given a worker's observed career history, each system ranks candidate next occupations.

The project compares four families of signals:

1. Dense semantic retrieval with pretrained and task-adapted encoders.
2. Train-only transition statistics fused with dense retrievers.
3. Frozen occupational Wiki knowledge and a transition KG.
4. Query-time Qwen reranking/routing ablations.

The cleaned repository is organized so that the final paper results, teacher-requested checks, and reproducibility scripts are separate from older exploratory outputs.

## Repository layout

| Path | Contents |
| --- | --- |
| `paper/` | Compiled manuscript PDF and framework figure files. |
| `src/` | Evaluation, analysis, data export, Wiki/KG, and worked-example scripts. |
| `training/` | Retriever training handoff code for Sentence Transformer, Late Interaction, BGE-M3/FlagEmbedding, Qwen3 embedding, and shared utilities. |
| `results/formal_eval/` | Compact formal evaluation summary table used by the paper. |
| `results/paper_checks/` | Final checks requested during review: formal BGE/ST per-query exports, paired significance, error analysis with BGE, filter counts, provenance checks, LateOn check, and worked-example search. |
| `results/wiki_qwen/` | Wiki/KG test results and Qwen validation ablations. |
| `data/full_metadata/` | Metadata for the exported full JobHop split, not the full raw dataset. |
| `environment/` | Docker, build, and Slurm helper files. |
| `docs/` | Human-readable notes explaining which results to use and how the teacher TODOs were addressed. |
| `scripts/` | Lightweight helper scripts for smoke checks and reproduction guidance. |

## Main final results

The formal test split contains 32,089 evaluation cases. The fixed candidate vocabulary contains 2,982 ESCO occupations.

| System | Split | MRR@10 | NDCG@10 | Recall@10 | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| BGE-M3 pure | test | 0.0685 | 0.0890 | 0.1554 | Formal per-query export. |
| BGE-M3 + Markov | test | 0.1397 | 0.1790 | 0.3071 | Markov alpha 0.5. |
| Trained Sentence Transformer pure | test | 0.1109 | 0.1493 | 0.2750 | Formal per-query export. |
| Trained Sentence Transformer + Markov | test | 0.1547 | 0.1982 | 0.3398 | Markov alpha 0.25. |
| Trained Qwen3 Embedding + Markov | test | 0.1516 | 0.1947 | 0.3356 | Formal evaluation summary. |
| LateOn + Markov | test | 0.1332 | 0.1720 | 0.2985 | Formal evaluation summary. |
| KG-alone | test | 0.1331 | 0.1761 | 0.3174 | Wiki/KG ablation. |
| Wiki+KG score fusion | test | 0.1377 | 0.1813 | 0.3239 | Fixed score fusion, alpha 0.8. |

Use the formal BGE outputs in `results/paper_checks/bge_m3_original_formal_per_query_full_test/`. Earlier `data_full` BGE checks were diagnostic and were superseded by the formal export.

## Teacher-requested checks

The review-facing checks are indexed in:

- `docs/TODO_CLOSURE_SUMMARY.md`
- `docs/WHICH_RESULTS_TO_USE.md`
- `results/RESULTS_INDEX.md`

The most important additions are:

- Formal BGE per-query export and paired significance.
- Formal trained Sentence Transformer per-query export and paired significance against BGE-M3 fusion.
- Error analysis extended with BGE-M3 pure/fusion and trained ST pure/fusion.
- Per-filter exact counts for train, validation, and test splits.
- Provenance checks for KG parameters, Wiki+KG alpha, Qwen failure handling, and candidate recall.
- Worked-example search outputs for a non-return case where Wiki+KG improves the gold rank.

## Reproducibility notes

The repository includes code and final outputs, but does not vendor large model weights or trained checkpoints. Public base models are downloaded from Hugging Face when required. Training scripts and configurations for task-adapted models are included under `training/`, and the submitted repository contains the final metrics and experimental outputs used in the accompanying paper.

Full reruns require:

- JobHop v2 data access and the export pipeline.
- Public base models such as `BAAI/bge-m3`.
- Trained model checkpoints regenerated from `training/` or restored externally.
- A Python/Docker environment matching `requirements.txt` and `run_docker.sh`.

For a quick repository integrity check:

```bash
bash scripts/smoke_test.sh
```

For reproduction guidance:

```bash
bash scripts/reproduce_main.sh
```

### Reproducing a representative main experiment

A representative main experiment that can be reproduced with public resources is the formal BGE-M3 + Markov experiment. It downloads JobHop v2 through Hugging Face datasets, loads BGE-M3, builds the train-derived ESCO candidate vocabulary and Markov transition statistics, selects the fusion weight on validation data, and evaluates the frozen configuration on the test split.

Requirements:

- NVIDIA GPU with CUDA.
- Internet access for Hugging Face datasets/models.
- Python dependencies from `requirements.txt`.

Run:

```bash
pip install -r requirements.txt

python src/export_bge_m3_formal_per_query.py \
  --model-name BAAI/bge-m3 \
  --cache-dir ./hf-cache \
  --output-dir ./outputs/reproduction_bge_m3
```

Expected test MRR@10 values are approximately:

- BGE-M3 pure: 0.069
- BGE-M3 + Markov: 0.140

Some secondary analysis scripts consume intermediate per-case files generated during the full experimental pipeline. The final secondary analysis outputs are included under `results/paper_checks/`.

### Submission archive

For a course submission ZIP, exclude repository metadata and redundant large intermediate files such as `.git/`, `__pycache__/`, and selected per-query JSONL exports. The GitHub repository keeps the curated result evidence, while the submission archive can be smaller and easier to inspect. A Windows helper is provided:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/make_submission_zip.ps1
```

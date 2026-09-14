#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cat <<'EOF'
This repository includes final paper outputs and the scripts that produced them.

The most directly reproducible main experiment is the formal BGE-M3 + Markov
experiment. It uses public Hugging Face resources and does not require a trained
project checkpoint.

Requirements:

- NVIDIA GPU with CUDA.
- Internet access for Hugging Face datasets/models.
- Python dependencies from requirements.txt.

Representative command:

pip install -r requirements.txt

python src/export_bge_m3_formal_per_query.py \
  --model-name BAAI/bge-m3 \
  --cache-dir ./hf-cache \
  --output-dir ./outputs/reproduction_bge_m3

Expected test MRR@10 values:

- BGE-M3 pure: about 0.069
- BGE-M3 + Markov: about 0.140

To execute this command from this helper, run:

RUN_BGE=1 bash scripts/reproduce_main.sh

Full reruns of all paper systems require additional external inputs:

1. The full exported JobHop data directory.
2. Public base models, for example BAAI/bge-m3.
3. Trained checkpoints for the task-adapted retrievers.
4. The Docker/Python environment described by requirements.txt and run_docker.sh.

Final paper outputs to verify first:

- results/formal_eval/formal_eval_summary.csv
- results/paper_checks/bge_m3_original_formal_per_query_full_test/
- results/paper_checks/trained_st_formal_per_query_full_test/
- results/paper_checks/significance_tests_fixed/
- results/paper_checks/error_analysis_plus_bge/
- results/paper_checks/filter_counts/
- results/paper_checks/provenance_checks_fast/
- results/wiki_qwen/summary/

Representative rerun commands once full data/checkpoints are restored:

python src/filter_counts_formal.py
python src/export_bge_m3_formal_per_query.py
python src/export_trained_st_formal_per_query_and_sig.py
python src/error_analysis_plus_bge.py
python src/jobhop_wiki_kg_ablation.py
python src/jobhop_wiki_kg_score_fusion.py

Consult each script's --help or run_config.json files in results/ for exact environment-specific paths.
EOF

if [[ "${RUN_BGE:-0}" == "1" ]]; then
  python src/export_bge_m3_formal_per_query.py \
    --model-name BAAI/bge-m3 \
    --cache-dir ./hf-cache \
    --output-dir ./outputs/reproduction_bge_m3
fi

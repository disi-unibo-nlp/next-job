#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cat <<'EOF'
This repository includes final paper outputs and the scripts that produced them.

Full reruns require large external inputs that are not committed:

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


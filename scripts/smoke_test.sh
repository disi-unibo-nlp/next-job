#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== repository smoke test =="

python -m py_compile src/*.py
echo "Python syntax check passed."

echo
echo "== required final result directories =="
for path in \
  results/formal_eval \
  results/paper_checks/bge_m3_original_formal_per_query_full_test \
  results/paper_checks/trained_st_formal_per_query_full_test \
  results/paper_checks/error_analysis_plus_bge \
  results/paper_checks/filter_counts \
  results/paper_checks/provenance_checks_fast \
  results/paper_checks/worked_example_search \
  results/wiki_qwen/summary
do
  if [[ -d "$path" ]]; then
    echo "OK  $path"
  else
    echo "MISSING  $path"
    exit 1
  fi
done

echo
echo "Smoke test complete."


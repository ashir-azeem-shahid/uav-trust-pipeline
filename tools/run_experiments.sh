#!/usr/bin/env bash
# Runs the experiments that need no broker, in the pinned environment:
#   1. the leakage tests            (ground truth never reaches a feature)
#   2. fault injection              (data quality layer vs benign faults)
#   3. unsupervised anomaly detection (Isolation Forest, clean baseline only)
#
# Usage (from the repo root, venv created):  bash tools/run_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "== 1/3 leakage tests =="
python -m pytest tests/ -q

echo
echo "== 2/3 fault injection =="
python tools/fault_inject.py

echo "== 3/3 anomaly detection =="
python tools/anomaly.py

echo "Done. Results: results_fault_injection.json  results_anomaly.json"

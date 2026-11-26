#!/usr/bin/env bash
# Runs the experiments that need no broker, in the pinned environment:
#   1. the leakage tests            (ground truth never reaches a feature)
#   2. fault injection              (data quality layer vs benign faults)
#   3. unsupervised anomaly detection (Isolation Forest, clean baseline only,
#      observer windows -- the comparison with the previous thesis)
#   4. the live ML models (train_ml.py): evaluation + models/ml.joblib
#
# Usage (from the repo root, venv created):  bash tools/run_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "== 1/4 leakage tests =="
python -m pytest tests/ -q

echo
echo "== 2/4 fault injection =="
python tools/fault_inject.py

echo "== 3/4 anomaly detection =="
python tools/anomaly.py

echo "== 4/4 live ML models =="
python train_ml.py

echo "Done. Results: results_fault_injection.json  results_anomaly.json  results_ml.json"

#!/usr/bin/env bash
set -euo pipefail
cd /home/user/PR-TTA
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
PYTHON=/home/user/Conda/CDF_FFT/bin/python
"$PYTHON" -m pytest -q tests/
"$PYTHON" -u experiments/e2b_estimator_training.py --effective
"$PYTHON" -u experiments/e2b_direction_comparison.py --phase effective_development
"$PYTHON" -u experiments/e2b_direction_comparison.py --phase gate
echo 'All E2b experiments completed' > logs/e2b_experiments.done
"$PYTHON" tools/audit_e2b.py
"$PYTHON" tools/report_e2b.py
echo 'Reports and audit completed' > logs/e2b_deliverables.done

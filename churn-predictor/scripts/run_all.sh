#!/usr/bin/env bash
# Run the full churn-predictor pipeline end-to-end, in order.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/01_download_and_validate_rct.py
python scripts/02_build_features.py
python scripts/03_baseline_response_model.py
python scripts/04_t_learner.py
python scripts/05_x_learner.py
python scripts/06_causal_forest.py
python scripts/07_evaluate_uplift.py
python scripts/08_revenue_simulation.py
python scripts/09_summarize_results.py

#!/bin/bash
set -euo pipefail
CONDA_ENV="prompt-pipeline"
REPO_ROOT="${SLURM_SUBMIT_DIR:?}"
cd "$REPO_ROOT"
source ~/.bashrc
conda activate "$CONDA_ENV"
export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONIOENCODING=utf-8
echo "--- Cross-backbone comparison (sd21 vs sdxl) ---"
python -m evaluation.backbone_compare outputs || echo "WARN: comparison failed (non-fatal)."

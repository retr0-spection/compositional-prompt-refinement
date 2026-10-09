#!/bin/bash
#SBATCH --job-name=rq5_cliptraj
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --nodelist=mscluster106,mscluster107
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/rq5_cliptraj_%j.out
#SBATCH --error=logs/slurm/rq5_cliptraj_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate prompt-pipeline
export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONIOENCODING=utf-8
echo "node: $(hostname)"; nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
python experiments/rq5_clip_trajectory.py

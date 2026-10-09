#!/bin/bash
#SBATCH --job-name=rq5_rescore
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm/rescore_%j.out
#SBATCH --error=logs/slurm/rescore_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate prompt-pipeline
export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONIOENCODING=utf-8
echo "node: $(hostname)"; nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
python experiments/rescore_relations.py

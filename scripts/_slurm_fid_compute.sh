#!/bin/bash
#SBATCH --job-name=fid_cmp
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --nodelist=mscluster106,mscluster107
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/fid_cmp_%j.out
#SBATCH --error=logs/slurm/fid_cmp_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate prompt-pipeline
export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONIOENCODING=utf-8
echo "node: $(hostname)"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
python experiments/compute_fid.py /datasets/onailana/coco/val2017

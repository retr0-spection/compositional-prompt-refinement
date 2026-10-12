#!/bin/bash
#SBATCH --job-name=fid_gen_bw
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --nodelist=mscluster110,mscluster111
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/fid_gen_bw_%j.out
#SBATCH --error=logs/slurm/fid_gen_bw_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate pp-bw   # CUDA 12.8 / sm_120 env for Blackwell nodes
export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONIOENCODING=utf-8
echo "node: $(hostname)"; nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
python experiments/generate_for_fid.py

#!/bin/bash
#SBATCH --job-name=fid_gen_bw
#SBATCH --partition=biggpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=mscluster110,mscluster111
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/fid_gen_bw_%j.out
#SBATCH --error=logs/slurm/fid_gen_bw_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
echo "job nodes: ${SLURM_JOB_NODELIST}"
# one task per Blackwell node; each shards the work via SLURM_PROCID/SLURM_NTASKS
srun bash -c 'cd "$SLURM_SUBMIT_DIR"; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pp-bw; export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"; export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONIOENCODING=utf-8; echo "task ${SLURM_PROCID} on $(hostname)"; nvidia-smi --query-gpu=name,memory.free --format=csv,noheader | head -1; python experiments/generate_for_fid.py'

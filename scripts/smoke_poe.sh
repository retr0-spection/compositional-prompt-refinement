#!/bin/bash
#SBATCH --job-name=poe-smoke
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/poe_smoke_%j.out
#SBATCH --error=logs/slurm/poe_smoke_%j.err

# Smoke-test the PoE positional-alignment fix: does compose() now produce
# coherent text instead of comma-salad? Reads nothing, writes to the log.
set -euo pipefail
CONDA_ENV="prompt-pipeline"
REPO_ROOT="${SLURM_SUBMIT_DIR:?}"
cd "$REPO_ROOT"
source ~/.bashrc
conda activate "$CONDA_ENV"
export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONIOENCODING=utf-8

echo "=== PoE smoke test | job $SLURM_JOB_ID | node $(hostname) ==="
nvidia-smi

python - <<'PYEOF'
from rewriters.llada_rewriter import LLaDARewriter
rw = LLaDARewriter()
tests = [
    ["a red cat", "a blue dog", "the cat is left of the dog"],
    ["a green book", "a yellow cup", "cup above book"],
    ["a small wooden chair", "a large metal table", "chair beside table"],
]
for c in tests:
    print("\nCONSTRAINTS:", c)
    print("COMPOSED   :", repr(rw.compose(c)))
print("\n=== smoke test done ===")
PYEOF

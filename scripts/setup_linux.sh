#!/bin/bash
#SBATCH --job-name=prompt-setup
#SBATCH --output=/home-mscluster/onailana/logs/setup_%j.txt
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00

# =============================================================================
# scripts/setup_linux.sh
#
# Full environment setup for the SLURM cluster.
#
# Usage
# -----
#   1. Fill in .env with your tokens (do this from the login node first):
#        cp .env.example .env
#        nano .env   # set WANDB_API_KEY and HF_TOKEN
#
#   2. Submit to SLURM from the repo root:
#        sbatch scripts/setup_linux.sh
#
#   3. Watch the log:
#        tail -f /home-mscluster/onailana/logs/setup_<jobid>.txt
#
# Edit the #SBATCH lines above and CONDA_ENV below to match your cluster.
# Update --output to your home directory path.
#
# Optional: pass a CUDA tag as arg to force a specific PyTorch build:
#   sbatch scripts/setup_linux.sh cu121
#   sbatch scripts/setup_linux.sh cu118
#   sbatch scripts/setup_linux.sh cpu
# =============================================================================

CONDA_ENV="prompt-pipeline"
CUDA_TAG="${1:-auto}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source ~/.bashrc

# ------------------------------
# CUDA / PyTorch config
# ------------------------------
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONFAULTHANDLER=1

echo "========================================"
echo "Job      : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "GPUs     : $CUDA_VISIBLE_DEVICES"
echo "Conda    : $CONDA_ENV"
echo "Repo     : $REPO_ROOT"
echo "========================================"
nvidia-smi

# Load credentials from .env
set -a; [[ -f .env ]] && source .env; set +a

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN not set in .env — gated models (LLaDA, SD 2.1) will fail to download."
fi

# ---------------------------------------------------------------------------
# 0. Detect CUDA version (GPU is available on compute node)
# ---------------------------------------------------------------------------
if [[ "$CUDA_TAG" == "auto" ]]; then
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
        MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        CUDA_TAG=$( [[ "$MAJOR" -ge 12 ]] && echo "cu121" || echo "cu118" )
        echo "[0] nvcc: CUDA $CUDA_VER → $CUDA_TAG"
    else
        DRIVER_CUDA=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)
        if [[ -n "$DRIVER_CUDA" ]]; then
            MAJOR=$(echo "$DRIVER_CUDA" | cut -d. -f1)
            CUDA_TAG=$( [[ "$MAJOR" -ge 12 ]] && echo "cu121" || echo "cu118" )
            echo "[0] nvidia-smi: CUDA $DRIVER_CUDA → $CUDA_TAG"
        else
            echo "[0] No CUDA detected — CPU-only PyTorch. LLaDA will not run."
            CUDA_TAG="cpu"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 1. Conda environment
# ---------------------------------------------------------------------------
echo ""
if conda env list | grep -qE "^${CONDA_ENV}\s"; then
    echo "[1/6] Conda env '${CONDA_ENV}' already exists — skipping creation."
else
    echo "[1/6] Creating conda env '${CONDA_ENV}' (Python 3.12)..."
    conda create -n "$CONDA_ENV" python=3.12 -y
fi

conda activate "$CONDA_ENV"
echo "  Python: $(python --version)"

# ---------------------------------------------------------------------------
# 2. PyTorch
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Installing PyTorch ($CUDA_TAG)..."
if [[ "$CUDA_TAG" == "cpu" ]]; then
    pip install --quiet torch torchvision
else
    pip install --quiet torch torchvision \
        --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

python - <<'PYCHECK'
import torch
print(f"  PyTorch {torch.__version__}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"  CUDA: {torch.cuda.get_device_name(0)} ({props.total_memory / 1e9:.1f} GB)")
else:
    print("  CUDA not available (CPU mode)")
PYCHECK

# ---------------------------------------------------------------------------
# 3. pip dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Installing dependencies from requirements.txt..."
pip install --quiet -r requirements.txt

# ---------------------------------------------------------------------------
# 4. Ollama (AR baseline) — installed INTO the conda env via conda-forge.
#    No root needed; binary lands at $CONDA_PREFIX/bin/ollama on the shared
#    filesystem, so every compute node that activates the env can find it.
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Setting up Ollama (conda-forge)..."
if ! command -v ollama &>/dev/null || [[ "$(command -v ollama)" != "$CONDA_PREFIX"* ]]; then
    echo "  Installing ollama into env '${CONDA_ENV}'..."
    conda install -n "$CONDA_ENV" -c conda-forge ollama -y
fi
echo "  Ollama: $(command -v ollama) ($(ollama --version 2>/dev/null || echo 'installed'))"

# Model weights go to shared home — visible from all nodes.
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
mkdir -p "$OLLAMA_MODELS"

ollama serve &>/tmp/ollama_setup.log &
OLLAMA_PID=$!
OLLAMA_READY=false
for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags &>/dev/null; then
        OLLAMA_READY=true
        break
    fi
    sleep 2
done
if ! $OLLAMA_READY; then
    echo "  ERROR: Ollama failed to start. Log:" >&2
    cat /tmp/ollama_setup.log >&2
    exit 1
fi
echo "  Ollama ready. Pre-pulling llama3.1..."
ollama pull llama3.1
echo "  llama3.1 ready."
kill "$OLLAMA_PID" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 5. HuggingFace + W&B login
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Auth setup..."
if [[ -n "${HF_TOKEN:-}" ]]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null \
        && echo "  HF login OK." \
        || echo "  WARNING: HF login failed — check HF_TOKEN in .env"
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    python -c "import wandb; wandb.login(key='${WANDB_API_KEY}', relogin=True)" 2>/dev/null \
        && echo "  W&B login OK." || true
fi

# ---------------------------------------------------------------------------
# 6. Smoke test
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Running smoke test..."
python test_pipeline.py

echo ""
echo "========================================"
echo "  Setup complete."
echo "  Activate with:  conda activate ${CONDA_ENV}"
echo "  Submit jobs:    bash scripts/submit_hpc.sh"
echo "========================================"

#!/usr/bin/env bash
# =============================================================================
# scripts/setup_linux.sh
#
# Run from the login node. Collects credentials interactively, then
# auto-submits itself to a GPU compute node via SLURM for installation
# and verification (GPU is not available on login nodes).
#
# Usage
# -----
#   bash scripts/setup_linux.sh              # auto-detect CUDA version
#   bash scripts/setup_linux.sh cu121        # force CUDA 12.1
#   bash scripts/setup_linux.sh cu118        # force CUDA 11.8
#   bash scripts/setup_linux.sh cpu          # CPU-only (smoke tests only)
#
# What it does
# ------------
#   Login node  : collects W&B and HuggingFace credentials → saves to .env
#                 → submits this script as a SLURM batch job
#   SLURM node  : detects CUDA, creates venv, installs PyTorch + deps,
#                 installs Ollama, pre-pulls llama3.1, runs smoke test
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Cluster resource settings — edit to match your cluster
# ---------------------------------------------------------------------------
PARTITION="gpu"
GPU_TYPE="a100"         # GPU constraint label (blank = any GPU)
CPUS_PER_TASK=8
MEM="32G"
TIME="02:00:00"         # venv + pip install + Ollama pull + smoke test

LOG_DIR="$REPO_ROOT/logs/slurm"
mkdir -p "$LOG_DIR"

CUDA_TAG="${1:-auto}"

# ===========================================================================
# PHASE 1 — Login node: collect credentials, then sbatch ourselves
# ===========================================================================
if [[ -z "${SLURM_JOB_ID:-}" ]]; then

    echo "============================================================"
    echo "  Prompt Pipeline — Setup (login node)"
    echo "  This script will submit itself to a GPU node via SLURM."
    echo "  Collecting credentials first (interactive)."
    echo "============================================================"

    # ---- .env bootstrap ----
    if [[ ! -f .env ]]; then
        cat > .env << 'EOF'
WANDB_API_KEY=
WANDB_OFFLINE=false
HF_TOKEN=
EOF
    fi
    set -a; source .env; set +a

    # ---- W&B ----
    echo ""
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        echo "W&B API key already set in .env — skipping."
    else
        read -rp "Enter your W&B API key (leave blank for offline mode): " WANDB_KEY
        if [[ -n "$WANDB_KEY" ]]; then
            sed -i "s|^WANDB_API_KEY=.*|WANDB_API_KEY=${WANDB_KEY}|" .env
            echo "  Key saved to .env"
        else
            sed -i "s|^WANDB_OFFLINE=.*|WANDB_OFFLINE=true|" .env
            echo "  Offline mode enabled."
        fi
    fi

    # ---- HuggingFace ----
    echo ""
    if [[ -n "${HF_TOKEN:-}" ]]; then
        echo "HF_TOKEN already set in .env — skipping."
    else
        read -rp "Enter your HuggingFace token (huggingface.co/settings/tokens, leave blank to skip): " HF_KEY
        if [[ -n "$HF_KEY" ]]; then
            sed -i "s|^HF_TOKEN=.*|HF_TOKEN=${HF_KEY}|" .env
            echo "  Token saved to .env"
        else
            echo "  Skipping HF token. Gated models (LLaDA, SD 2.1) will fail to download."
        fi
    fi

    # ---- Submit self as SLURM batch job ----
    echo ""
    echo "Submitting setup job to SLURM partition '${PARTITION}'..."

    CONSTRAINT_ARG=""
    [[ -n "$GPU_TYPE" ]] && CONSTRAINT_ARG="--constraint=${GPU_TYPE}"

    JOB_ID=$(sbatch \
        --job-name="prompt-setup" \
        --partition="${PARTITION}" \
        ${CONSTRAINT_ARG:+"$CONSTRAINT_ARG"} \
        --gres=gpu:1 \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --mem="${MEM}" \
        --time="${TIME}" \
        --output="${LOG_DIR}/setup_%j.out" \
        --error="${LOG_DIR}/setup_%j.err" \
        "$0" "${CUDA_TAG}" \
        | grep -oP '\d+')

    echo ""
    echo "============================================================"
    echo "  Setup job submitted: $JOB_ID"
    echo ""
    echo "  Monitor progress:"
    echo "    squeue -j $JOB_ID"
    echo "    tail -f ${LOG_DIR}/setup_${JOB_ID}.out"
    echo ""
    echo "  Once complete, activate your environment with:"
    echo "    source venv/bin/activate"
    echo "  Then submit experiments:"
    echo "    bash scripts/submit_hpc.sh"
    echo "============================================================"
    exit 0
fi

# ===========================================================================
# PHASE 2 — SLURM compute node: install and verify
# ===========================================================================

echo "============================================================"
echo "  Prompt Pipeline — Setup (SLURM node)"
echo "  Job      : $SLURM_JOB_ID"
echo "  Node     : $(hostname)"
echo "  GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'none')"
echo "  CUDA tag : $CUDA_TAG"
echo "  Directory: $REPO_ROOT"
echo "============================================================"

# Load .env so tokens are available to pip/huggingface-cli/wandb
set -a; [[ -f .env ]] && source .env; set +a

# ---------------------------------------------------------------------------
# 0. Detect CUDA version (GPU is available here)
# ---------------------------------------------------------------------------
if [[ "$CUDA_TAG" == "auto" ]]; then
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
        MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        if [[ "$MAJOR" -ge 12 ]]; then
            CUDA_TAG="cu121"
        elif [[ "$MAJOR" -eq 11 && "$MINOR" -ge 8 ]]; then
            CUDA_TAG="cu118"
        else
            echo "WARNING: CUDA $CUDA_VER detected but only cu121/cu118 supported. Defaulting to cu121."
            CUDA_TAG="cu121"
        fi
        echo "[0] Detected CUDA $CUDA_VER → using $CUDA_TAG"
    else
        # nvcc absent but nvidia-smi may still exist (driver-only install)
        DRIVER_CUDA=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)
        if [[ -n "$DRIVER_CUDA" ]]; then
            MAJOR=$(echo "$DRIVER_CUDA" | cut -d. -f1)
            CUDA_TAG=$( [[ "$MAJOR" -ge 12 ]] && echo "cu121" || echo "cu118" )
            echo "[0] nvcc not found; driver reports CUDA $DRIVER_CUDA → using $CUDA_TAG"
        else
            echo "[0] WARNING: No CUDA detected — installing CPU-only PyTorch. LLaDA will not run."
            CUDA_TAG="cpu"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 1. Python 3.12 virtual environment
# ---------------------------------------------------------------------------
echo ""
if [[ ! -d venv ]]; then
    echo "[1/6] Creating virtual environment with Python 3.12..."
    if command -v python3.12 &>/dev/null; then
        python3.12 -m venv venv
    else
        echo "ERROR: python3.12 not found. Load the module first, e.g.:"
        echo "  module load python/3.12"
        exit 1
    fi
else
    echo "[1/6] Virtual environment already exists — skipping creation."
fi
source venv/bin/activate

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
    print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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
# 4. Ollama (AR baseline)
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Setting up Ollama..."
if ! command -v ollama &>/dev/null; then
    echo "  Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "  Ollama installed."
else
    echo "  Ollama already installed: $(ollama --version 2>/dev/null || echo 'unknown version')"
fi

# Start Ollama in background and wait for it
if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
    echo "  Starting Ollama server..."
    ollama serve &>/tmp/ollama_setup.log &
    OLLAMA_PID=$!
    for i in $(seq 1 30); do
        curl -sf http://localhost:11434/api/tags &>/dev/null && break
        sleep 2
    done
    echo "  Ollama ready (PID $OLLAMA_PID)."
fi

echo "  Pre-pulling llama3.1 (this may take a few minutes)..."
ollama pull llama3.1
echo "  llama3.1 ready."

# ---------------------------------------------------------------------------
# 5. HuggingFace login (token already in .env from Phase 1)
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] HuggingFace login..."
if [[ -n "${HF_TOKEN:-}" ]]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null \
        && echo "  HF login OK." \
        || echo "  WARNING: huggingface-cli login failed — check your token in .env"
else
    echo "  No HF_TOKEN in .env — skipping. Gated models will fail to download."
fi

# W&B login
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
echo "============================================================"
echo "  Setup complete."
echo "  Activate:     source venv/bin/activate"
echo "  Submit jobs:  bash scripts/submit_hpc.sh"
echo "============================================================"

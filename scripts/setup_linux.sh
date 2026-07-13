#!/usr/bin/env bash
# =============================================================================
# scripts/setup_linux.sh
#
# Full environment setup for a Linux HPC node (SLURM cluster or bare metal).
# Run once before submitting jobs.
#
# Usage
# -----
#   bash scripts/setup_linux.sh             # auto-detect CUDA version
#   bash scripts/setup_linux.sh cu121       # force CUDA 12.1
#   bash scripts/setup_linux.sh cu118       # force CUDA 11.8
#   bash scripts/setup_linux.sh cpu         # CPU-only (smoke tests only)
#
# What it does
# ------------
#   1. Creates a Python 3.12 virtual environment in ./venv
#   2. Installs PyTorch with the correct CUDA index URL
#   3. Installs all pip dependencies from requirements.txt
#   4. Installs Ollama (for the AR rewrite baseline)
#   5. Prompts for a W&B API key and saves it to .env
#   6. Runs the smoke test to verify the stack
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 0. Detect CUDA version
# ---------------------------------------------------------------------------
CUDA_TAG="${1:-auto}"

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
            echo "WARNING: CUDA $CUDA_VER detected but only cu121/cu118 are supported. Defaulting to cu121."
            CUDA_TAG="cu121"
        fi
    else
        echo "WARNING: nvcc not found — installing CPU-only PyTorch. LLaDA will not run."
        CUDA_TAG="cpu"
    fi
fi

echo "============================================================"
echo "  Prompt Pipeline — Linux Setup"
echo "  CUDA tag : $CUDA_TAG"
echo "  Directory: $REPO_ROOT"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Python 3.12 virtual environment
# ---------------------------------------------------------------------------
if [[ ! -d venv ]]; then
    echo ""
    echo "[1/7] Creating virtual environment with Python 3.12..."
    if command -v python3.12 &>/dev/null; then
        python3.12 -m venv venv
    else
        echo "ERROR: python3.12 not found. Install it first:"
        echo "  Ubuntu/Debian: sudo apt install python3.12 python3.12-venv"
        echo "  Or via pyenv:  pyenv install 3.12.x && pyenv local 3.12.x"
        exit 1
    fi
else
    echo ""
    echo "[1/7] Virtual environment already exists — skipping creation."
fi

source venv/bin/activate

# ---------------------------------------------------------------------------
# 2. PyTorch
# ---------------------------------------------------------------------------
echo ""
echo "[2/7] Installing PyTorch ($CUDA_TAG)..."
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
else:
    print("  CUDA not available (CPU mode)")
PYCHECK

# ---------------------------------------------------------------------------
# 3. pip dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[3/7] Installing dependencies from requirements.txt..."
pip install --quiet -r requirements.txt

# ---------------------------------------------------------------------------
# 4. Ollama (AR baseline)
# ---------------------------------------------------------------------------
echo ""
echo "[4/7] Installing Ollama..."
if command -v ollama &>/dev/null; then
    echo "  Ollama already installed: $(ollama --version 2>/dev/null || echo 'unknown version')"
else
    curl -fsSL https://ollama.com/install.sh | sh
    echo "  Ollama installed."
fi

# Start Ollama in background if not running
if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
    echo "  Starting Ollama server in background..."
    ollama serve &>/tmp/ollama.log &
    sleep 3
    echo "  Ollama PID: $!"
fi

# Pre-pull llama3.1 so the first experiment run doesn't wait
echo "  Pre-pulling llama3.1 (this may take a few minutes)..."
ollama pull llama3.1

# ---------------------------------------------------------------------------
# 5. W&B API key
# ---------------------------------------------------------------------------
echo ""
echo "[5/7] Weights & Biases setup..."
if [[ -f .env ]] && grep -q "WANDB_API_KEY" .env; then
    echo "  W&B API key already set in .env — skipping."
else
    read -rp "  Enter your W&B API key (leave blank to use offline mode): " WANDB_KEY
    if [[ -n "$WANDB_KEY" ]]; then
        echo "WANDB_API_KEY=${WANDB_KEY}" >> .env
        echo "  Key saved to .env"
        python -c "import wandb; wandb.login(key='${WANDB_KEY}')" 2>/dev/null && echo "  W&B login verified." || true
    else
        echo "  Skipping W&B key — experiments will run with --no-wandb or offline mode."
        grep -q "WANDB_OFFLINE" .env 2>/dev/null || echo "WANDB_OFFLINE=true" >> .env
    fi
fi

# ---------------------------------------------------------------------------
# 6. Hugging Face login
# ---------------------------------------------------------------------------
echo ""
echo "[6/7] Hugging Face setup..."
# Load .env so we can check for an existing token
set -a; [[ -f .env ]] && source .env; set +a

if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "  HF_TOKEN found in .env — logging in..."
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null \
        && echo "  HF login OK." \
        || echo "  WARNING: huggingface-cli login failed — check your token."
else
    read -rp "  Enter your HuggingFace token (from huggingface.co/settings/tokens, leave blank to skip): " HF_KEY
    if [[ -n "$HF_KEY" ]]; then
        # Persist to .env
        if grep -q "^HF_TOKEN=" .env 2>/dev/null; then
            sed -i "s|^HF_TOKEN=.*|HF_TOKEN=${HF_KEY}|" .env
        else
            echo "HF_TOKEN=${HF_KEY}" >> .env
        fi
        huggingface-cli login --token "$HF_KEY" --add-to-git-credential 2>/dev/null \
            && echo "  HF login OK — token saved to .env" \
            || echo "  WARNING: login failed — check your token."
    else
        echo "  Skipping HF login. Public models will still download; gated models will fail."
    fi
fi

# ---------------------------------------------------------------------------
# 7. Smoke test
# ---------------------------------------------------------------------------
echo ""
echo "Running smoke test..."
# Reload env vars (token may have just been written)
set -a; [[ -f .env ]] && source .env; set +a

python test_pipeline.py

echo ""
echo "============================================================"
echo "  Setup complete."
echo "  Activate with:  source venv/bin/activate"
echo "  Submit jobs:    bash scripts/submit_hpc.sh"
echo "============================================================"

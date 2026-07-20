#!/bin/bash
# =============================================================================
# scripts/submit_hpc.sh
#
# Submit all experiment jobs to SLURM.
#
# Job graph
# ---------
#
#   [warmup_ar]   [warmup_llada]          ← Job 0: populate rewrite caches
#        \              /
#         afterok  afterok
#              \  /
#       ┌──────────────────┐
#       │  RQ1 array (×3)  │              ← text + embedding analysis
#       │  RQ2 array (×3)  │              ← image gen + scoring
#       │  RQ3 array (×3)  │              ← CFG sweep
#       │  RQ4 array (×3)  │              ← AR vs LLaDA
#       └──────────────────┘
#
# Array task index → pipeline:
#   0  raw_clip
#   1  ar_clip
#   2  llada_clip
#
# (Long-CLIP disabled pending EmbeddingProjector fine-tuning.)
#
# Usage
# -----
#   bash scripts/submit_hpc.sh              # submit everything
#   bash scripts/submit_hpc.sh --rq 2      # warmup + RQ2 only
#   bash scripts/submit_hpc.sh --dry-run   # print sbatch commands, don't submit
#
# Requirements
# ------------
#   - SLURM environment with sbatch/squeue
#   - Ollama installed in the conda env:  conda install -c conda-forge ollama
#     (weights pulled once from a login node: ollama pull llama3.1)
#   - HF_TOKEN and WANDB_API_KEY set in .env
#   - Edit the resource parameters below to match your cluster
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Cluster resource settings — edit for your partition and hardware
# ---------------------------------------------------------------------------
CONDA_ENV="prompt-pipeline" # conda env name — must match setup_linux.sh
PARTITION="biggpu"          # less contention than bigbatch; no --gres needed

# Memory per job — sized to actual model footprints, not --mem=0
MEM_WARMUP_AR="8G"      # Ollama + llama3.1
MEM_WARMUP_LLADA="20G"  # LLaDA-8B-Instruct (bfloat16, ~16 GB weights)
MEM_RQ1="4G"            # CLIP encoding only, all rewrites are cache hits
MEM_RQ2="16G"           # two-phase: SD 2.1 then BLIP-2, never simultaneous
MEM_RQ3="8G"            # CFG sweep, no BLIP-2
MEM_RQ4="16G"           # same two-phase scorer stack as RQ2

TIME_WARMUP_AR="04:00:00"   # AR warmup: ~442 Ollama calls, sequential
TIME_WARMUP_LLADA="12:00:00" # LLaDA warmup: ~442 × 128-step masked diffusion passes
TIME_RQ1="01:00:00"         # RQ1: text + embeddings, all cache hits → fast
TIME_RQ2="12:00:00"         # RQ2: image gen (50 steps × 500 prompts) + BLIP-2 scoring
TIME_RQ3="06:00:00"         # RQ3: CFG sweep over 25 prompts × 5 scales
TIME_RQ4="08:00:00"         # RQ4: AR vs LLaDA head-to-head

VENV="$REPO_ROOT/venv"
LOG_DIR="$REPO_ROOT/logs/slurm"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
SUBMIT_RQS=("1" "2" "3" "4")
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rq)      SUBMIT_RQS=("$2"); shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Load .env (W&B key, HF token, offline flag)
# ---------------------------------------------------------------------------
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a; source "$REPO_ROOT/.env"; set +a
fi
WANDB_API_KEY="${WANDB_API_KEY:-}"
HF_TOKEN="${HF_TOKEN:-}"

# ---------------------------------------------------------------------------
# Active pipeline array (3 conditions; Long-CLIP disabled)
# ---------------------------------------------------------------------------
PIPELINE_NAMES=("raw_clip" "ar_clip" "llada_clip")
N_PIPELINES=${#PIPELINE_NAMES[@]}   # 3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_sbatch() {
    # Wraps sbatch; in dry-run mode prints the command instead of submitting.
    # Returns the numeric job ID (or "DRY_RUN" in dry-run mode).
    if $DRY_RUN; then
        echo "[DRY-RUN] sbatch $*" >&2
        echo "DRY_RUN"
    else
        sbatch "$@" | grep -oP '\d+'
    fi
}

_node_args() {
    echo \
        "--partition=${PARTITION}" \
        "--nodes=1" \
        "--cpus-per-task=16"
}

_common_env() {
    # Variables exported to every job
    echo \
        "WANDB_API_KEY=${WANDB_API_KEY}" \
        "HF_TOKEN=${HF_TOKEN}" \
        "WANDB_PROJECT=prompt-pipeline" \
        "SEED=42"
}

# ---------------------------------------------------------------------------
# Generate shared task scripts (written once, reused by all array tasks)
# ---------------------------------------------------------------------------

# ---- Main RQ task (RQ1-4 array jobs) ----
TASK_SCRIPT="$REPO_ROOT/scripts/_slurm_task.sh"
cat > "$TASK_SCRIPT" << 'TASK_EOF'
#!/bin/bash
# Auto-generated by submit_hpc.sh — do not edit.
set -euo pipefail
CONDA_ENV="prompt-pipeline"

# sbatch copies this script to the slurmd spool dir, so BASH_SOURCE cannot be
# used to locate the repo. SLURM_SUBMIT_DIR = directory sbatch was invoked
# from, which submit_hpc.sh guarantees is the repo root.
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set — submit via sbatch from repo root}"
cd "$REPO_ROOT"

source ~/.bashrc
conda activate "$CONDA_ENV"

set -a; [[ -f "${REPO_ROOT}/.env" ]] && source "${REPO_ROOT}/.env"; set +a

# ------------------------------
# CUDA / PyTorch config
# ------------------------------
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONFAULTHANDLER=1

echo "========================================"
echo "Job      : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "Repo     : $REPO_ROOT"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "RQ       : $RQ"
echo "========================================"
nvidia-smi

[[ -n "${WANDB_API_KEY:-}" ]] && \
    python -c "import wandb; wandb.login(key='${WANDB_API_KEY}', relogin=True)" 2>/dev/null || true

# Run all pipelines sequentially in this single job
PIPELINE_NAMES=("raw_clip" "ar_clip" "llada_clip")
for PIPELINE_NAME in "${PIPELINE_NAMES[@]}"; do
    echo ""
    echo "--- Pipeline: $PIPELINE_NAME ---"
    python experiments/run_experiment.py \
        --rq "$RQ" \
        --pipeline "$PIPELINE_NAME" \
        --seed "${SEED:-42}"
done
echo "All pipelines complete for RQ${RQ}."
TASK_EOF
chmod +x "$TASK_SCRIPT"

# ---- AR warmup task (CPU node, needs Ollama) ----
WARMUP_AR_SCRIPT="$REPO_ROOT/scripts/_slurm_warmup_ar.sh"
cat > "$WARMUP_AR_SCRIPT" << 'WARMUP_AR_EOF'
#!/bin/bash
# Auto-generated by submit_hpc.sh — do not edit.
# Warm the AR rewrite cache: Ollama + Llama 3.1 over all experiment prompts.
# GPU node so Ollama can offload llama3.1 layers to VRAM (~4 GB).
set -euo pipefail
CONDA_ENV="prompt-pipeline"

# sbatch copies this script to the slurmd spool dir — use SLURM_SUBMIT_DIR.
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set — submit via sbatch from repo root}"
cd "$REPO_ROOT"

source ~/.bashrc
conda activate "$CONDA_ENV"

set -a; [[ -f "${REPO_ROOT}/.env" ]] && source "${REPO_ROOT}/.env"; set +a

# ------------------------------
# Locate Ollama.
# Primary: the conda env binary ($CONDA_PREFIX/bin/ollama), installed via
#   conda install -c conda-forge ollama
# conda activate above puts it on PATH, so command -v resolves it.
# OLLAMA_BIN in .env overrides everything (e.g. module-based installs).
# ------------------------------
if [[ -z "${OLLAMA_BIN:-}" ]]; then
    for candidate in \
        "${CONDA_PREFIX:-}/bin/ollama" \
        "$(command -v ollama 2>/dev/null || true)" \
        "$HOME/bin/ollama" \
        "$HOME/.local/bin/ollama"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            OLLAMA_BIN="$candidate"
            break
        fi
    done
fi
if [[ -z "${OLLAMA_BIN:-}" || ! -x "${OLLAMA_BIN}" ]]; then
    echo "FATAL: ollama binary not found on $(hostname)." >&2
    echo "Install into the env:  conda activate ${CONDA_ENV} && conda install -c conda-forge ollama" >&2
    echo "Or set OLLAMA_BIN=/path/to/ollama in .env" >&2
    exit 1
fi
echo "Using Ollama: $OLLAMA_BIN ($("$OLLAMA_BIN" --version 2>/dev/null || echo 'version unknown'))"

# Model weights live on the shared home filesystem — visible to all nodes.
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"

# ------------------------------
# CUDA / PyTorch config
# ------------------------------
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONFAULTHANDLER=1

echo "========================================"
echo "Job      : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "Phase    : AR rewrite cache warmup (ar_clip)"
echo "========================================"
nvidia-smi

# Start Ollama in the background and wait for it to be ready.
"$OLLAMA_BIN" serve &
OLLAMA_PID=$!
echo "Ollama PID: $OLLAMA_PID"

# Poll until /api/tags responds (up to 60 s) — FAIL HARD on timeout.
OLLAMA_READY=false
for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        OLLAMA_READY=true
        break
    fi
    # Bail out early if the server process already died
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "FATAL: Ollama process died during startup." >&2
        exit 1
    fi
    sleep 2
done
if ! $OLLAMA_READY; then
    echo "FATAL: Ollama did not become ready within 60 s." >&2
    kill "$OLLAMA_PID" 2>/dev/null || true
    exit 1
fi
echo "Ollama ready."

# Verify the model is actually pulled — fail with a clear message, not a
# cryptic 404 from the rewriter mid-run.
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1}"
if ! curl -sf http://localhost:11434/api/tags | grep -q "\"${OLLAMA_MODEL}"; then
    echo "FATAL: model '${OLLAMA_MODEL}' not found in Ollama." >&2
    echo "Pull it once from a login node:" >&2
    echo "  conda activate ${CONDA_ENV} && ollama serve & ollama pull ${OLLAMA_MODEL}" >&2
    kill "$OLLAMA_PID" 2>/dev/null || true
    exit 1
fi
echo "Model '${OLLAMA_MODEL}' available."

python experiments/run_experiment.py \
    --rq 0 \
    --pipeline ar_clip \
    --no-wandb

echo "AR warmup complete."
kill "$OLLAMA_PID" 2>/dev/null || true
WARMUP_AR_EOF
chmod +x "$WARMUP_AR_SCRIPT"

# ---- LLaDA warmup task (GPU node) ----
WARMUP_LLADA_SCRIPT="$REPO_ROOT/scripts/_slurm_warmup_llada.sh"
cat > "$WARMUP_LLADA_SCRIPT" << 'WARMUP_LLADA_EOF'
#!/bin/bash
# Auto-generated by submit_hpc.sh — do not edit.
# Warm the LLaDA rewrite cache: LLaDA-8B-Instruct over all experiment prompts.
set -euo pipefail
CONDA_ENV="prompt-pipeline"

# sbatch copies this script to the slurmd spool dir — use SLURM_SUBMIT_DIR.
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set — submit via sbatch from repo root}"
cd "$REPO_ROOT"

source ~/.bashrc
conda activate "$CONDA_ENV"

set -a; [[ -f "${REPO_ROOT}/.env" ]] && source "${REPO_ROOT}/.env"; set +a

# ------------------------------
# CUDA / PyTorch config
# ------------------------------
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONFAULTHANDLER=1

echo "========================================"
echo "Job      : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "Phase    : LLaDA rewrite cache warmup (llada_clip)"
echo "========================================"
nvidia-smi

[[ -n "${WANDB_API_KEY:-}" ]] && \
    python -c "import wandb; wandb.login(key='${WANDB_API_KEY}', relogin=True)" 2>/dev/null || true

python experiments/run_experiment.py \
    --rq 0 \
    --pipeline llada_clip \
    --no-wandb

echo "LLaDA warmup complete."
WARMUP_LLADA_EOF
chmod +x "$WARMUP_LLADA_SCRIPT"

# ---------------------------------------------------------------------------
# Step 0: Warmup — skip if cache files already exist
# ---------------------------------------------------------------------------
CACHE_DIR="$REPO_ROOT/outputs/rewrite_cache"
AR_CACHE="${CACHE_DIR}/ar_llama3.1.json"
LLADA_CACHE="${CACHE_DIR}/llada.json"

if [[ -f "$AR_CACHE" ]] && [[ -f "$LLADA_CACHE" ]]; then
    echo ""
    echo "=== Rewrite cache already populated — skipping warmup jobs ==="
    WARMUP_AR_ID="(cached)"
    WARMUP_LLADA_ID="(cached)"
    DEPENDENCY=""
else
    echo ""
    echo "=== Step 0a: AR rewrite warmup ==="
    WARMUP_AR_ID=$(_sbatch \
        $(_node_args) \
        --mem="${MEM_WARMUP_AR}" \
        --time="${TIME_WARMUP_AR}" \
        --job-name="prompt-warmup-ar" \
        --output="${LOG_DIR}/warmup_ar_%j.out" \
        --error="${LOG_DIR}/warmup_ar_%j.err" \
        "$WARMUP_AR_SCRIPT")
    echo "  Job ID: $WARMUP_AR_ID"

    echo ""
    echo "=== Step 0b: LLaDA rewrite warmup ==="
    WARMUP_LLADA_ID=$(_sbatch \
        $(_node_args) \
        --mem="${MEM_WARMUP_LLADA}" \
        --time="${TIME_WARMUP_LLADA}" \
        --job-name="prompt-warmup-llada" \
        --output="${LOG_DIR}/warmup_llada_%j.out" \
        --error="${LOG_DIR}/warmup_llada_%j.err" \
        "$WARMUP_LLADA_SCRIPT")
    echo "  Job ID: $WARMUP_LLADA_ID"

    if $DRY_RUN; then
        DEPENDENCY="afterok:WARMUP_AR_ID:WARMUP_LLADA_ID"
    else
        DEPENDENCY="afterok:${WARMUP_AR_ID}:${WARMUP_LLADA_ID}"
    fi
fi

# ---------------------------------------------------------------------------
# Step 1-4: Submit RQ jobs sequentially chained (one job per RQ)
# Each RQ waits for the previous one to finish before starting.
# Pipelines run sequentially inside each job — no arrays.
# ---------------------------------------------------------------------------
declare -A RQ_JOB_IDS
PREV_DEPENDENCY="$DEPENDENCY"   # first RQ depends on warmup; subsequent RQs chain off each other

for RQ in "${SUBMIT_RQS[@]}"; do
    case "$RQ" in
        1) TIME_LIMIT=$TIME_RQ1; MEM_LIMIT=$MEM_RQ1 ;;
        2) TIME_LIMIT=$TIME_RQ2; MEM_LIMIT=$MEM_RQ2 ;;
        3) TIME_LIMIT=$TIME_RQ3; MEM_LIMIT=$MEM_RQ3 ;;
        4) TIME_LIMIT=$TIME_RQ4; MEM_LIMIT=$MEM_RQ4 ;;
        *) echo "Unknown RQ: $RQ"; exit 1 ;;
    esac

    echo ""
    echo "=== RQ${RQ} (pipelines: ${PIPELINE_NAMES[*]}, sequential) ==="

    DEP_ARG=""
    [[ -n "$PREV_DEPENDENCY" ]] && DEP_ARG="--dependency=${PREV_DEPENDENCY}"

    JOB_ID=$(_sbatch \
        $(_node_args) \
        --mem="${MEM_LIMIT}" \
        --time="${TIME_LIMIT}" \
        --job-name="prompt-rq${RQ}" \
        ${DEP_ARG:+"$DEP_ARG"} \
        --output="${LOG_DIR}/rq${RQ}_%j.out" \
        --error="${LOG_DIR}/rq${RQ}_%j.err" \
        --export=ALL,RQ="${RQ}",SEED=42,WANDB_PROJECT=prompt-pipeline \
        "$TASK_SCRIPT")

    RQ_JOB_IDS[$RQ]="$JOB_ID"
    [[ -n "$PREV_DEPENDENCY" ]] \
        && echo "  Job ID: $JOB_ID  (depends on: $PREV_DEPENDENCY)" \
        || echo "  Job ID: $JOB_ID  (no dependency — starts immediately)"

    # Next RQ chains off this one
    if $DRY_RUN; then
        PREV_DEPENDENCY="afterok:RQ${RQ}_JOB_ID"
    else
        PREV_DEPENDENCY="afterok:${JOB_ID}"
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Warmup AR    : ${WARMUP_AR_ID}"
echo "  Warmup LLaDA : ${WARMUP_LLADA_ID}"
for RQ in "${SUBMIT_RQS[@]}"; do
    echo "  RQ${RQ}          : ${RQ_JOB_IDS[$RQ]:-n/a}"
done
echo ""
echo "  Monitor:"
echo "    squeue -u \$USER"
echo "    tail -f ${LOG_DIR}/warmup_llada_*.out"
echo "    tail -f ${LOG_DIR}/rq2_*.out"
echo ""
echo "  Cancel all submitted jobs:"
if ! $DRY_RUN; then
    ALL_IDS="${WARMUP_AR_ID} ${WARMUP_LLADA_ID}"
    for RQ in "${SUBMIT_RQS[@]}"; do
        ALL_IDS+=" ${RQ_JOB_IDS[$RQ]:-}"
    done
    echo "    scancel ${ALL_IDS}"
fi
echo "============================================================"

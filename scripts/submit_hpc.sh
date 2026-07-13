#!/usr/bin/env bash
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
#   - Ollama installed in PATH on all GPU/CPU nodes (for AR rewriter)
#   - HF_TOKEN and WANDB_API_KEY set in .env
#   - Edit the resource parameters below to match your cluster
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Cluster resource settings — edit for your partition and hardware
# ---------------------------------------------------------------------------
PARTITION_GPU="gpu"         # Partition with GPU nodes (LLaDA warmup + RQ2/4)
PARTITION_CPU="cpu"         # Partition for CPU-only jobs (AR warmup; set = PARTITION_GPU if no separate CPU partition)
GPU_TYPE="a100"             # GPU constraint label (blank = any GPU)
CPUS_PER_TASK=8
MEM_GPU="64G"               # Memory for GPU nodes
MEM_CPU="32G"               # Memory for CPU-only AR warmup

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

_gpu_args() {
    local constraint_arg=""
    [[ -n "$GPU_TYPE" ]] && constraint_arg="--constraint=${GPU_TYPE}"
    echo \
        "--partition=${PARTITION_GPU}" \
        ${constraint_arg:+"$constraint_arg"} \
        "--gres=gpu:1" \
        "--cpus-per-task=${CPUS_PER_TASK}" \
        "--mem=${MEM_GPU}"
}

_cpu_args() {
    echo \
        "--partition=${PARTITION_CPU}" \
        "--cpus-per-task=${CPUS_PER_TASK}" \
        "--mem=${MEM_CPU}"
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
#!/usr/bin/env bash
# Auto-generated by submit_hpc.sh — do not edit.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source "${REPO_ROOT}/venv/bin/activate"
[[ -f "${REPO_ROOT}/.env" ]] && set -a && source "${REPO_ROOT}/.env" && set +a

# Resolve pipeline name from array index
PIPELINE_NAMES=("raw_clip" "ar_clip" "llada_clip")
PIPELINE_NAME="${PIPELINE_NAMES[$SLURM_ARRAY_TASK_ID]}"

echo "========================================"
echo "Job      : $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "Node     : $(hostname)"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'none')"
echo "RQ       : $RQ"
echo "Pipeline : $PIPELINE_NAME"
echo "========================================"

[[ -n "${WANDB_API_KEY:-}" ]] && \
    python -c "import wandb; wandb.login(key='${WANDB_API_KEY}', relogin=True)" 2>/dev/null || true

python experiments/run_experiment.py \
    --rq "$RQ" \
    --pipeline "$PIPELINE_NAME" \
    --seed "${SEED:-42}"
TASK_EOF
chmod +x "$TASK_SCRIPT"

# ---- AR warmup task (CPU node, needs Ollama) ----
WARMUP_AR_SCRIPT="$REPO_ROOT/scripts/_slurm_warmup_ar.sh"
cat > "$WARMUP_AR_SCRIPT" << 'WARMUP_AR_EOF'
#!/usr/bin/env bash
# Auto-generated by submit_hpc.sh — do not edit.
# Warm the AR rewrite cache: runs Ollama + Llama 3.1 over all experiment prompts.
# Submitted to a GPU node so Ollama can offload llama3.1 layers to VRAM (~4 GB).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source "${REPO_ROOT}/venv/bin/activate"
[[ -f "${REPO_ROOT}/.env" ]] && set -a && source "${REPO_ROOT}/.env" && set +a

echo "========================================"
echo "Job      : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'none')"
echo "Phase    : AR rewrite cache warmup (ar_clip)"
echo "========================================"

# Start Ollama in the background and wait for it to be ready.
ollama serve &
OLLAMA_PID=$!
echo "Ollama PID: $OLLAMA_PID"

# Poll until /api/tags responds (up to 60 s)
for i in $(seq 1 30); do
    curl -sf http://localhost:11434/api/tags > /dev/null 2>&1 && break
    sleep 2
done
echo "Ollama ready."

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
#!/usr/bin/env bash
# Auto-generated by submit_hpc.sh — do not edit.
# Warm the LLaDA rewrite cache: runs LLaDA-8B-Instruct over all experiment prompts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source "${REPO_ROOT}/venv/bin/activate"
[[ -f "${REPO_ROOT}/.env" ]] && set -a && source "${REPO_ROOT}/.env" && set +a

echo "========================================"
echo "Job      : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'none')"
echo "Phase    : LLaDA rewrite cache warmup (llada_clip)"
echo "========================================"

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
# Step 0: Submit warmup jobs (run in parallel, no dependency)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 0a: AR rewrite warmup (GPU node, Ollama + Llama 3.1) ==="
WARMUP_AR_ID=$(_sbatch \
    $(_gpu_args) \
    --time="${TIME_WARMUP_AR}" \
    --job-name="prompt-warmup-ar" \
    --output="${LOG_DIR}/warmup_ar_%j.out" \
    --error="${LOG_DIR}/warmup_ar_%j.err" \
    "$WARMUP_AR_SCRIPT")
echo "  Job ID: $WARMUP_AR_ID"

echo ""
echo "=== Step 0b: LLaDA rewrite warmup (GPU node) ==="
WARMUP_LLADA_ID=$(_sbatch \
    $(_gpu_args) \
    --time="${TIME_WARMUP_LLADA}" \
    --job-name="prompt-warmup-llada" \
    --output="${LOG_DIR}/warmup_llada_%j.out" \
    --error="${LOG_DIR}/warmup_llada_%j.err" \
    "$WARMUP_LLADA_SCRIPT")
echo "  Job ID: $WARMUP_LLADA_ID"

# Build SLURM dependency string (both warmup jobs must succeed)
if $DRY_RUN; then
    DEPENDENCY="afterok:WARMUP_AR_ID:WARMUP_LLADA_ID"
else
    DEPENDENCY="afterok:${WARMUP_AR_ID}:${WARMUP_LLADA_ID}"
fi

# ---------------------------------------------------------------------------
# Step 1-4: Submit RQ array jobs, dependent on both warmup jobs
# ---------------------------------------------------------------------------
declare -A RQ_JOB_IDS

for RQ in "${SUBMIT_RQS[@]}"; do
    case "$RQ" in
        1) TIME_LIMIT=$TIME_RQ1 ;;
        2) TIME_LIMIT=$TIME_RQ2 ;;
        3) TIME_LIMIT=$TIME_RQ3 ;;
        4) TIME_LIMIT=$TIME_RQ4 ;;
        *) echo "Unknown RQ: $RQ"; exit 1 ;;
    esac

    echo ""
    echo "=== RQ${RQ} array (${N_PIPELINES} tasks: ${PIPELINE_NAMES[*]}) ==="

    JOB_ID=$(_sbatch \
        $(_gpu_args) \
        --time="${TIME_LIMIT}" \
        --job-name="prompt-rq${RQ}" \
        --array="0-$((N_PIPELINES-1))" \
        --dependency="${DEPENDENCY}" \
        --output="${LOG_DIR}/rq${RQ}_%A_%a.out" \
        --error="${LOG_DIR}/rq${RQ}_%A_%a.err" \
        --export=ALL,RQ="${RQ}",SEED=42,WANDB_PROJECT=prompt-pipeline,WANDB_API_KEY="${WANDB_API_KEY}",HF_TOKEN="${HF_TOKEN}" \
        "$TASK_SCRIPT")

    RQ_JOB_IDS[$RQ]="$JOB_ID"
    echo "  Job ID: $JOB_ID  (depends on warmup jobs $WARMUP_AR_ID + $WARMUP_LLADA_ID)"
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Warmup AR    : ${WARMUP_AR_ID}"
echo "  Warmup LLaDA : ${WARMUP_LLADA_ID}"
for RQ in "${SUBMIT_RQS[@]}"; do
    echo "  RQ${RQ} array    : ${RQ_JOB_IDS[$RQ]:-n/a}"
done
echo ""
echo "  Monitor:"
echo "    squeue -u \$USER"
echo "    tail -f ${LOG_DIR}/warmup_llada_*.out   # longest job"
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

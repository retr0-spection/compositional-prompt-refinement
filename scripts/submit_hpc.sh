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
MEM_RQ5="20G"           # PoE: LLaDA + SD + BLIP-2 (small subset)
MEM_RQ6="16G"           # tunability: LLaDA + SD + CLIPScore, no BLIP-2

TIME_WARMUP_AR="04:00:00"   # AR warmup: ~442 Ollama calls, sequential
TIME_WARMUP_LLADA="12:00:00" # LLaDA warmup: ~442 × 128-step masked diffusion passes
TIME_RQ1="01:00:00"         # RQ1: text + embeddings, all cache hits → fast
TIME_RQ2="12:00:00"         # RQ2: image gen (50 steps × 500 prompts) + BLIP-2 scoring
TIME_RQ3="06:00:00"         # RQ3: CFG sweep over 25 prompts × 5 scales
TIME_RQ4="08:00:00"         # RQ4: AR vs LLaDA head-to-head
TIME_RQ5="06:00:00"         # RQ5: PoE capstone, ~15-20 prompts x 4 conditions
TIME_RQ6="12:00:00"         # RQ6: 256-cell tunability grid x 15 prompts

# ---------------------------------------------------------------------------
# Backbone-aware resource sizing.
# The above limits are tuned for SD 2.1 @ 768². SDXL @ 1024² is ~2-4x slower
# per image and uses more host RAM (dual encoders + larger buffers), so raise
# the generation-heavy RQ limits when the config selects the sdxl backbone.
# Rewrite warmups are backbone-agnostic (they populate caches) and unchanged.
# This reads config/experiment.yaml so `backbone: sdxl` adjusts the model AND
# the SLURM budget from one switch. Only ever RAISES limits; sd21 is untouched.
# ---------------------------------------------------------------------------
if grep -qE '^\s*backbone:\s*sdxl' "${REPO_ROOT}/config/experiment.yaml" 2>/dev/null; then
    echo "Detected SDXL backbone — applying larger time/memory budgets."
    TIME_RQ2="24:00:00"
    TIME_RQ3="12:00:00"
    TIME_RQ4="18:00:00"
    MEM_RQ2="24G"
    MEM_RQ3="12G"
    MEM_RQ4="24G"
fi

VENV="$REPO_ROOT/venv"
LOG_DIR="$REPO_ROOT/logs/slurm"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
SUBMIT_RQS=("1" "2" "3" "4" "5" "6")
DRY_RUN=false
# Backbones to run in the SAME chain, in order. Each backbone runs the full
# RQ sequence writing to outputs/<backbone>/. Override with --backbones.
BACKBONES=("sd21" "sdxl")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rq)        SUBMIT_RQS=("$2"); shift 2 ;;
        --backbones) IFS=',' read -ra BACKBONES <<< "$2"; shift 2 ;;
        --dry-run)   DRY_RUN=true; shift ;;
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
# Compute nodes run with an unset locale (ASCII default), which makes Python's
# open()/print() crash on non-ASCII chars (em-dashes, LLaDA output). Force UTF-8.
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8

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

# Run pipelines according to RQ needs:
#   RQ1-3: each pipeline is independent, run them as separate processes
#          (lets a single failure not kill the others; summaries merge via
#           per-pipeline JSON sidecars).
#   RQ4-6: run once in ONE process (RQ4 = mechanism comparison needs both AR+
#          LLaDA; RQ5 = PoE builds its own 4 conditions; RQ6 = tunability grid).
#
# BACKBONE (env var, default sd21) selects the T2I backbone. We override ALL
# backbone-coupled config fields together (not just backbone) so the yaml's
# committed SDXL model_id/resolution/prediction_type don't leak into an sd21
# run — that mismatch loads SDXL weights into the SD2.1 pipeline and crashes
# with 'added_cond_kwargs is None'. run_experiment.py scopes output to
# outputs/<backbone>/ automatically.
BACKBONE="${BACKBONE:-sd21}"
echo "Backbone: $BACKBONE"
if [[ "$BACKBONE" == "sdxl" ]]; then
    BACKBONE_OVERRIDE=(--config t2i.backbone=sdxl
                       t2i.model_id=stabilityai/stable-diffusion-xl-base-1.0
                       t2i.resolution=1024 t2i.prediction_type=epsilon)
else
    BACKBONE_OVERRIDE=(--config t2i.backbone=sd21
                       t2i.model_id=sd2-community/stable-diffusion-2-1
                       t2i.resolution=768 t2i.prediction_type=v_prediction)
fi

if [[ "$RQ" == "4" || "$RQ" == "5" || "$RQ" == "6" ]]; then
    echo ""
    echo "--- RQ${RQ}: single-process run (backbone=$BACKBONE) ---"
    python experiments/run_experiment.py \
        --rq "$RQ" \
        --seed "${SEED:-42}" \
        "${BACKBONE_OVERRIDE[@]}"
else
    PIPELINE_NAMES=("raw_clip" "ar_clip" "llada_clip")
    for PIPELINE_NAME in "${PIPELINE_NAMES[@]}"; do
        echo ""
        echo "--- Pipeline: $PIPELINE_NAME (backbone=$BACKBONE) ---"
        python experiments/run_experiment.py \
            --rq "$RQ" \
            --pipeline "$PIPELINE_NAME" \
            --seed "${SEED:-42}" \
            "${BACKBONE_OVERRIDE[@]}"
    done
fi
echo "All pipelines complete for RQ${RQ} (backbone=$BACKBONE)."

# Regenerate all plots from disk artifacts, scoped to this backbone's outputs.
echo ""
echo "--- Regenerating plots (outputs/${BACKBONE}) ---"
python -m evaluation.plotting "outputs/${BACKBONE}" || echo "WARN: plotting step failed (non-fatal)."
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
# Primary: official tarball install at ~/ollama-dist (bin/ + lib/ollama/).
# NOTE: conda-forge/homebrew ollama packages are BROKEN for GGUF models —
# they omit the llama-server runner, which ollama looks up via hardcoded
# paths relative to its own binary. The tarball ships the full layout:
#   mkdir -p ~/ollama-dist
#   curl -L https://ollama.com/download/ollama-linux-amd64.tgz | tar -xz -C ~/ollama-dist
# OLLAMA_BIN in .env overrides everything.
# ------------------------------
if [[ -z "${OLLAMA_BIN:-}" ]]; then
    for candidate in \
        "$HOME/ollama-dist/bin/ollama" \
        "$HOME/bin/ollama" \
        "$HOME/.local/bin/ollama" \
        "$(command -v ollama 2>/dev/null || true)"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            OLLAMA_BIN="$candidate"
            break
        fi
    done
fi
if [[ -z "${OLLAMA_BIN:-}" || ! -x "${OLLAMA_BIN}" ]]; then
    echo "FATAL: ollama binary not found on $(hostname)." >&2
    echo "Install the official tarball (includes llama-server runner):" >&2
    echo "  mkdir -p ~/ollama-dist" >&2
    echo "  curl -L https://ollama.com/download/ollama-linux-amd64.tgz | tar -xz -C ~/ollama-dist" >&2
    exit 1
fi

# Verify the install is complete — catches broken conda/brew-style installs early.
# Two valid layouts:
#   - Ollama >= 0.2x: separate llama-server runner in lib/ollama/
#   - Ollama 0.1x (e.g. v0.13.5): integrated runner, ggml backends in lib/ollama/
# Broken packages ship the bare binary with NO lib/ollama contents at all.
OLLAMA_LIB_DIR="$(dirname "$OLLAMA_BIN")/../lib/ollama"
if [[ ! -e "$OLLAMA_LIB_DIR/llama-server" \
   && ! -e "$(dirname "$OLLAMA_BIN")/llama-server" \
   && -z "$(ls "$OLLAMA_LIB_DIR"/libggml-base.so* 2>/dev/null)" ]]; then
    echo "FATAL: $OLLAMA_BIN exists but no inference runtime found (no llama-server, no ggml libs)." >&2
    echo "This install cannot run GGUF models. Use the official bundle:" >&2
    echo "  curl -L https://github.com/ollama/ollama/releases/download/v0.13.5/ollama-linux-amd64.tgz | tar -xz -C ~/ollama-dist" >&2
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
# Step 1-6: Submit RQ jobs, looped over BACKBONES.
# Each backbone runs the full RQ sequence writing to outputs/<backbone>/.
# The chain is: warmup -> [sd21 RQ1..6] -> [sdxl RQ1..6], each RQ waiting on
# the previous. Backbones run sequentially (sdxl starts after sd21's last RQ)
# so they never contend for the same GPU or race on outputs.
# ---------------------------------------------------------------------------
declare -A RQ_JOB_IDS
PREV_DEPENDENCY="$DEPENDENCY"   # first job depends on warmup; rest chain

for BACKBONE in "${BACKBONES[@]}"; do
    echo ""
    echo "############################################################"
    echo "#  Backbone: ${BACKBONE}"
    echo "############################################################"

    for RQ in "${SUBMIT_RQS[@]}"; do
        case "$RQ" in
            1) TIME_LIMIT=$TIME_RQ1; MEM_LIMIT=$MEM_RQ1 ;;
            2) TIME_LIMIT=$TIME_RQ2; MEM_LIMIT=$MEM_RQ2 ;;
            3) TIME_LIMIT=$TIME_RQ3; MEM_LIMIT=$MEM_RQ3 ;;
            4) TIME_LIMIT=$TIME_RQ4; MEM_LIMIT=$MEM_RQ4 ;;
            5) TIME_LIMIT=$TIME_RQ5; MEM_LIMIT=$MEM_RQ5 ;;
            6) TIME_LIMIT=$TIME_RQ6; MEM_LIMIT=$MEM_RQ6 ;;
            *) echo "Unknown RQ: $RQ"; exit 1 ;;
        esac

        # SDXL is slower/heavier — raise limits for the generation-heavy RQs.
        if [[ "$BACKBONE" == "sdxl" ]]; then
            case "$RQ" in
                2) TIME_LIMIT="24:00:00"; MEM_LIMIT="24G" ;;
                3) TIME_LIMIT="12:00:00"; MEM_LIMIT="12G" ;;
                4) TIME_LIMIT="18:00:00"; MEM_LIMIT="24G" ;;
                5) TIME_LIMIT="08:00:00"; MEM_LIMIT="24G" ;;
                6) TIME_LIMIT="24:00:00"; MEM_LIMIT="24G" ;;
            esac
        fi

        echo ""
        echo "=== [${BACKBONE}] RQ${RQ} ==="

        DEP_ARG=""
        [[ -n "$PREV_DEPENDENCY" ]] && DEP_ARG="--dependency=${PREV_DEPENDENCY}"

        JOB_ID=$(_sbatch \
            $(_node_args) \
            --mem="${MEM_LIMIT}" \
            --time="${TIME_LIMIT}" \
            --job-name="prompt-${BACKBONE}-rq${RQ}" \
            ${DEP_ARG:+"$DEP_ARG"} \
            --output="${LOG_DIR}/${BACKBONE}_rq${RQ}_%j.out" \
            --error="${LOG_DIR}/${BACKBONE}_rq${RQ}_%j.err" \
            --export=ALL,RQ="${RQ}",BACKBONE="${BACKBONE}",SEED=42,WANDB_PROJECT=prompt-pipeline \
            "$TASK_SCRIPT")

        RQ_JOB_IDS["${BACKBONE}_${RQ}"]="$JOB_ID"
        [[ -n "$PREV_DEPENDENCY" ]] \
            && echo "  Job ID: $JOB_ID  (depends on: $PREV_DEPENDENCY)" \
            || echo "  Job ID: $JOB_ID  (starts immediately)"

        if $DRY_RUN; then
            PREV_DEPENDENCY="afterok:${BACKBONE}_RQ${RQ}_JOB_ID"
        else
            PREV_DEPENDENCY="afterok:${JOB_ID}"
        fi
    done
done

# ---------------------------------------------------------------------------
# Final step: cross-backbone comparison plots (reads outputs/sd21 + outputs/sdxl)
# Chained after the very last RQ job so both backbones are complete.
# ---------------------------------------------------------------------------
COMPARE_SCRIPT="$REPO_ROOT/scripts/_slurm_compare.sh"
cat > "$COMPARE_SCRIPT" << 'COMPARE_EOF'
#!/bin/bash
set -euo pipefail
CONDA_ENV="prompt-pipeline"
REPO_ROOT="${SLURM_SUBMIT_DIR:?}"
cd "$REPO_ROOT"
source ~/.bashrc
conda activate "$CONDA_ENV"
export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONIOENCODING=utf-8
echo "--- Cross-backbone comparison (sd21 vs sdxl) ---"
python -m evaluation.backbone_compare outputs || echo "WARN: comparison failed (non-fatal)."
COMPARE_EOF
chmod +x "$COMPARE_SCRIPT"

echo ""
echo "=== Cross-backbone comparison ==="
COMPARE_DEP=""
[[ -n "$PREV_DEPENDENCY" ]] && COMPARE_DEP="--dependency=${PREV_DEPENDENCY}"
COMPARE_ID=$(_sbatch \
    $(_node_args) \
    --mem="4G" \
    --time="00:30:00" \
    --job-name="prompt-compare" \
    ${COMPARE_DEP:+"$COMPARE_DEP"} \
    --output="${LOG_DIR}/compare_%j.out" \
    --error="${LOG_DIR}/compare_%j.err" \
    "$COMPARE_SCRIPT")
echo "  Job ID: $COMPARE_ID"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Warmup AR    : ${WARMUP_AR_ID}"
echo "  Warmup LLaDA : ${WARMUP_LLADA_ID}"
for BACKBONE in "${BACKBONES[@]}"; do
    for RQ in "${SUBMIT_RQS[@]}"; do
        echo "  [${BACKBONE}] RQ${RQ}    : ${RQ_JOB_IDS[${BACKBONE}_${RQ}]:-n/a}"
    done
done
echo "  Compare      : ${COMPARE_ID:-n/a}"
echo ""
echo "  Monitor:"
echo "    squeue -u \$USER"
echo "    tail -f ${LOG_DIR}/warmup_llada_*.out"
echo "    tail -f ${LOG_DIR}/rq2_*.out"
echo ""
echo "  Cancel all submitted jobs:"
if ! $DRY_RUN; then
    ALL_IDS="${WARMUP_AR_ID} ${WARMUP_LLADA_ID}"
    for BACKBONE in "${BACKBONES[@]}"; do
        for RQ in "${SUBMIT_RQS[@]}"; do
            ALL_IDS+=" ${RQ_JOB_IDS[${BACKBONE}_${RQ}]:-}"
        done
    done
    ALL_IDS+=" ${COMPARE_ID:-}"
    echo "    scancel ${ALL_IDS}"
fi
echo "============================================================"

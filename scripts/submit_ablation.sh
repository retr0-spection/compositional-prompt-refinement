#!/bin/bash
#SBATCH --job-name=poe-ablation
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/poe_ablation_%j.out
#SBATCH --error=logs/slurm/poe_ablation_%j.err
# =============================================================================
# scripts/submit_ablation.sh
#
# RQ5 PoE *construction* ablation (text-only, fast — no image generation).
#
# Compares the original single-constraint "disjoint" experts (which collapse
# into comma-salad on competing objects) against the "shared_scene" experts
# (each conditions on the whole scene, emphasising one constraint), with and
# without a whole-scene fluency-anchor expert. Reports, per construction, the
# comma-salad fraction and constraint coverage over the rq5_compositional set.
#
# This is the cheap iteration loop: pick a winning construction here, then run
# the full RQ5 image pipeline (submit_hpc.sh --rq 5) with it.
#
# Usage
# -----
#   sbatch scripts/submit_ablation.sh                 # all 18 prompts
#   LIMIT=4 sbatch scripts/submit_ablation.sh         # quick 4-prompt check
#   OUT=outputs/sdxl/rq5_ablation sbatch scripts/submit_ablation.sh
#
# Requirements: GPU (LLaDA-8B) and Ollama for scene-graph decomposition —
# same as an RQ5 run. Weights are assumed already pulled (see submit_hpc.sh).
# =============================================================================
set -euo pipefail
CONDA_ENV="prompt-pipeline"

# sbatch copies this script to the spool dir, so locate the repo via
# SLURM_SUBMIT_DIR (the dir sbatch was invoked from — run from repo root).
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set — submit via sbatch from repo root}"
cd "$REPO_ROOT"

source ~/.bashrc
conda activate "$CONDA_ENV"
set -a; [[ -f "${REPO_ROOT}/.env" ]] && source "${REPO_ROOT}/.env"; set +a

# ------------------------------ runtime config ------------------------------
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONFAULTHANDLER=1
# Compute nodes default to an ASCII locale, which crashes print()/open() on the
# non-ASCII text LLaDA emits. Force UTF-8 (same as the RQ task scripts).
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8

OUT="${OUT:-outputs/sdxl/rq5_ablation}"
LIMIT="${LIMIT:-0}"                       # 0 = all prompts
PROMPT_SET="${PROMPT_SET:-rq5_compositional}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1}"

echo "========================================"
echo "Job      : ${SLURM_JOB_ID:-<interactive>}"
echo "Node     : $(hostname)"
echo "Repo     : $REPO_ROOT"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "Out      : $OUT   (limit=$LIMIT, set=$PROMPT_SET)"
echo "========================================"
nvidia-smi || true

# ------------------------------ start Ollama --------------------------------
# The SemanticExtractor decomposes each prompt into constraints via the LLM
# backend, exactly as an RQ5 run does. Without it, decomposition silently falls
# back to keyword extraction and the ablation is run on weaker constraints.
OLLAMA_BIN="${OLLAMA_BIN:-$HOME/ollama-dist/bin/ollama}"
OLLAMA_PID=""
if [[ -x "$OLLAMA_BIN" ]]; then
    echo "Starting Ollama ($OLLAMA_BIN)..."
    export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
    "$OLLAMA_BIN" serve & OLLAMA_PID=$!
    OLLAMA_READY=false
    for i in $(seq 1 30); do
        if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
            OLLAMA_READY=true; break
        fi
        if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
            echo "FATAL: Ollama died during startup." >&2; exit 1
        fi
        sleep 2
    done
    $OLLAMA_READY || { echo "FATAL: Ollama not ready within 60 s." >&2; \
        kill "$OLLAMA_PID" 2>/dev/null || true; exit 1; }
    if ! curl -sf http://localhost:11434/api/tags | grep -q "\"${OLLAMA_MODEL}"; then
        echo "FATAL: model '${OLLAMA_MODEL}' not pulled. On a login node:" >&2
        echo "  conda activate ${CONDA_ENV} && ollama serve & ollama pull ${OLLAMA_MODEL}" >&2
        kill "$OLLAMA_PID" 2>/dev/null || true; exit 1
    fi
    echo "Ollama ready (PID $OLLAMA_PID), model '${OLLAMA_MODEL}' available."
else
    echo "FATAL: Ollama binary not found at $OLLAMA_BIN — needed for decomposition." >&2
    echo "Install the official tarball (see submit_hpc.sh) or set OLLAMA_BIN." >&2
    exit 1
fi

# ------------------------------ run the ablation ----------------------------
ABLATION_ARGS=(--prompt-set "$PROMPT_SET" --out "$OUT" --ollama-model "$OLLAMA_MODEL")
[[ "$LIMIT" != "0" ]] && ABLATION_ARGS+=(--limit "$LIMIT")

set +e
python -m experiments.rq5_poe_ablation "${ABLATION_ARGS[@]}"
STATUS=$?
set -e

[[ -n "$OLLAMA_PID" ]] && kill "$OLLAMA_PID" 2>/dev/null || true

if [[ $STATUS -eq 0 ]]; then
    echo ""
    echo "Ablation complete. Results:"
    echo "   $OUT/ablation.txt   (readable table + per-prompt outputs)"
    echo "   $OUT/ablation.json  (machine-readable)"
else
    echo "Ablation FAILED (exit $STATUS) — see the error above." >&2
fi
exit $STATUS

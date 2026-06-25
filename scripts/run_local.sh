#!/usr/bin/env bash
# =============================================================================
# scripts/run_local.sh
#
# Single-GPU dry run for local development — verifies the full stack quickly
# before committing GPU hours on the cluster.
#
# Usage
# -----
#   bash scripts/run_local.sh          # dry-run all RQs
#   bash scripts/run_local.sh --rq 2  # dry-run one RQ
# =============================================================================

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source venv/bin/activate
[[ -f .env ]] && set -a && source .env && set +a

RQ="${2:-all}"
[[ "$1" == "--rq" ]] && RQ="$2"

python experiments/run_experiment.py \
    --rq "$RQ" \
    --dry-run \
    --no-wandb \
    "$@"

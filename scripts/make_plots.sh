#!/bin/bash
# scripts/make_plots.sh
# Tier-1 plotting: regenerate all figures from data already on disk.
# No GPU needed — safe to run on the login node or locally.
#
# Usage:
#   bash scripts/make_plots.sh                 # both backbones + compare
#   bash scripts/make_plots.sh sd21            # one backbone only
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Which backbones to plot (default: both).
if [[ $# -gt 0 ]]; then
    BACKBONES=("$@")
else
    BACKBONES=("sd21" "sdxl")
fi

for BB in "${BACKBONES[@]}"; do
    if [[ -d "outputs/${BB}" ]]; then
        echo "=== Plotting outputs/${BB} ==="
        python -m evaluation.plotting "outputs/${BB}" \
            || echo "WARN: plotting failed for ${BB} (non-fatal)."
    else
        echo "--- Skipping ${BB}: outputs/${BB} not found ---"
    fi
done

echo ""
echo "=== Cross-backbone comparison (sd21 vs sdxl) ==="
python -m evaluation.backbone_compare outputs \
    || echo "WARN: backbone comparison failed (non-fatal)."

echo ""
echo "Done. Figures in outputs/<backbone>/plots and outputs/plots/compare."

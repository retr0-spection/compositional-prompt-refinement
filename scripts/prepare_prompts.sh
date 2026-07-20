#!/bin/bash
# =============================================================================
# scripts/prepare_prompts.sh
#
# One-time data preparation:
#   1. FID reference set — 500 real images from COCO val2017 into
#      data/fid_reference/, used by RQ2 for true (realism) FID.
#
# Network note: the cluster login node has restricted outbound access
# (ollama.com CDN stalls; github.com works). If the COCO download stalls,
# run this on a compute node instead:
#   sbatch --partition=bigbatch --mem=4G --time=01:00:00 \
#          --wrap="bash scripts/prepare_prompts.sh"
# or download val2017.zip on your local machine and scp it to /tmp/.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FID_DIR="data/fid_reference"
N_IMAGES=500
COCO_ZIP="/tmp/coco_val2017.zip"
COCO_URL="http://images.cocodataset.org/zips/val2017.zip"

echo "=== FID reference set (${N_IMAGES} COCO val2017 images) ==="

existing=$(find "$FID_DIR" -name '*.jpg' 2>/dev/null | wc -l || echo 0)
if [[ "$existing" -ge "$N_IMAGES" ]]; then
    echo "Already prepared: $existing images in $FID_DIR — skipping."
    exit 0
fi

mkdir -p "$FID_DIR"

if [[ ! -f "$COCO_ZIP" ]]; then
    echo "Downloading COCO val2017 (~1 GB)..."
    curl -fL --retry 3 --connect-timeout 30 "$COCO_URL" -o "$COCO_ZIP" || {
        echo "" >&2
        echo "FATAL: COCO download failed or stalled (cluster network restriction?)." >&2
        echo "Options:" >&2
        echo "  1. Run on a compute node:" >&2
        echo "     sbatch --partition=bigbatch --mem=4G --time=01:00:00 --wrap=\"bash scripts/prepare_prompts.sh\"" >&2
        echo "  2. Download on your local machine and scp to the cluster:" >&2
        echo "     curl -L $COCO_URL -o val2017.zip" >&2
        echo "     scp val2017.zip <user>@<cluster>:/tmp/coco_val2017.zip" >&2
        echo "     bash scripts/prepare_prompts.sh" >&2
        exit 1
    }
fi

echo "Extracting first ${N_IMAGES} images..."
# List zip entries, take the first N jpg paths, extract only those.
unzip -Z1 "$COCO_ZIP" | grep '\.jpg$' | sort | head -n "$N_IMAGES" > /tmp/coco_subset.txt
unzip -o -q "$COCO_ZIP" $(cat /tmp/coco_subset.txt) -d /tmp/coco_extract
mv /tmp/coco_extract/val2017/*.jpg "$FID_DIR/"
rm -rf /tmp/coco_extract /tmp/coco_subset.txt
rm -f "$COCO_ZIP"

final=$(find "$FID_DIR" -name '*.jpg' | wc -l)
echo "Done: $final reference images in $FID_DIR"
echo ""
echo "Ensure config/experiment.yaml contains:"
echo "  fid_reference_dir: data/fid_reference"

"""
evaluation/backbone_compare.py

Cross-backbone comparison: reads outputs/<backbone>/ for each backbone
(sd21, sdxl) and plots all metrics side by side, so the encoder-capacity
question — does a higher-capacity backbone change whether refinement helps? —
can be read directly.

This is the analysis layer for the SD 2.1 vs SDXL axis that replaced the
dropped Long-CLIP encoder comparison.

Reads the same disk artifacts the per-backbone plots use:
    outputs/<backbone>/rq2/<pipeline>/<set>/trace.jsonl      (per-prompt scores)

Output: outputs/plots/compare/*.png|pdf

Usage:
    python -m evaluation.backbone_compare outputs
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_BACKBONES = ["sd21", "sdxl"]
_BACKBONE_COLOURS = {"sd21": "#4C9F70", "sdxl": "#7B5EA7"}
_PIPELINE_ORDER = ["raw_clip", "ar_clip", "llada_clip"]


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False,
    })
    return plt


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
    logger.info("Wrote %s", out_dir / f"{name}.png")
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({f"compare/{name}": wandb.Image(str(out_dir / f"{name}.png"))})
    except Exception:
        pass
    import matplotlib.pyplot as plt
    plt.close(fig)


def _load_rq2_traces(backbone_root: Path) -> dict:
    """{pipeline: {set: [records]}} from one backbone's RQ2 traces."""
    rq2 = backbone_root / "rq2"
    out: dict = {}
    if not rq2.is_dir():
        return out
    for trace in sorted(rq2.glob("*/*/trace.jsonl")):
        pipe, set_name = trace.parent.parent.name, trace.parent.name
        try:
            recs = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines() if l.strip()]
            out.setdefault(pipe, {})[set_name] = recs
        except Exception as exc:
            logger.warning("Could not read %s (%s)", trace, exc)
    return out


def _applicable_mean(records, metric):
    if metric == "clip_score":
        vals = [r["clip_score"] for r in records if "clip_score" in r]
    elif metric == "attr":
        vals = [r["attr_binding"]["accuracy"] for r in records
                if r.get("attr_binding", {}).get("n_pairs", 0) > 0]
    elif metric == "relation":
        vals = [r["relation_accuracy"]["accuracy"] for r in records
                if r.get("relation_accuracy", {}).get("n_relations", 0) > 0]
    else:
        return None
    return sum(vals) / len(vals) if vals else None


def plot_backbone_metric(output_root: Path, metric: str) -> None:
    """
    Grouped bars: for each pipeline, SD 2.1 vs SDXL side by side, averaged
    over all RQ2 sets. Shows whether the backbone changes the refinement effect.
    """
    plt = _style()
    import numpy as np

    data = {}
    for bb in _BACKBONES:
        root = Path(output_root) / bb
        traces = _load_rq2_traces(root)
        if traces:
            data[bb] = traces
    if len(data) < 1:
        logger.info("No backbone data for metric %s — skipping.", metric)
        return

    pipes = _PIPELINE_ORDER
    x = np.arange(len(pipes))
    width = 0.8 / max(len(data), 1)

    fig, ax = plt.subplots(figsize=(1.8 * len(pipes) + 2, 4.5))
    for j, bb in enumerate([b for b in _BACKBONES if b in data]):
        ys = []
        for pipe in pipes:
            sets = data[bb].get(pipe, {})
            all_recs = [r for recs in sets.values() for r in recs]
            ys.append(_applicable_mean(all_recs, metric) if all_recs else None)
        xs = [x[k] + (j - len(data)/2 + 0.5) * width
              for k, y in enumerate(ys) if y is not None]
        yv = [y for y in ys if y is not None]
        ax.bar(xs, yv, width, label=bb, color=_BACKBONE_COLOURS.get(bb, "#333"))

    label = {"clip_score": "CLIPScore", "attr": "Attribute binding",
             "relation": "Relation accuracy"}[metric]
    ax.set_xticks(x)
    ax.set_xticklabels(pipes, rotation=15, ha="right")
    ax.set_ylabel(label)
    ax.set_title(f"SD 2.1 vs SDXL — {label} (pooled over RQ2 sets)")
    ax.legend(fontsize=10)
    _save(fig, Path(output_root) / "plots" / "compare", f"compare_{metric}")


def compare_all(output_root: str | Path = "outputs") -> None:
    """Render all cross-backbone comparison plots from disk."""
    root = Path(output_root)
    present = [bb for bb in _BACKBONES if (root / bb).is_dir()]
    if not present:
        logger.warning("No backbone output dirs found under %s — nothing to compare.", root)
        return
    logger.info("Comparing backbones: %s", present)
    for metric in ("clip_score", "attr", "relation"):
        plot_backbone_metric(root, metric)
    logger.info("Backbone comparison complete. Figures in %s", root / "plots" / "compare")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    compare_all(sys.argv[1] if len(sys.argv) > 1 else "outputs")

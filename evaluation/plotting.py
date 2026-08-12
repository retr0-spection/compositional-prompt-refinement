"""
evaluation/plotting.py

Publication-style plots for the full 2x3 experiment, reading the artifacts
the RQ runners already produce:

    outputs/rq2/<pipeline>/<set>/trace.jsonl     per-prompt RQ2 scores
    outputs/rq2/rq2_summary.txt                  (not parsed; trace is source of truth)
    outputs/rq3/<pipeline>/...                   CFG sweep (via rq3 result dicts)
    outputs/trajectory/<pipeline>/<set>/*.jsonl  per-denoising-step scores (if instrumented)

Two x-axes are supported:
  1. CFG scale   — from RQ3 sweep data (metric vs. guidance strength)
  2. Denoising timestep — from trajectory data (metric vs. diffusion step),
     only present if T2IRunner was run with a scoring callback (see
     generation/t2i_runner.py :: generate_with_trajectory).

Every figure is written as BOTH .png (raster, for slides) and .pdf (vector,
for the dissertation) and, if a W&B run is active, logged as an image panel.

Designed to run on partial data: pipelines/sets/scales that are absent are
skipped with a logged note, so this is usable now (llada_clip only) and
complete once all six arms finish.

NOTE ON DATA VALIDITY (read before trusting accuracy plots):
  - RQ2/RQ3 attribute & relation accuracy suffered a structural-zero
    aggregation bug (prompts with no attribute/relation counted as 0%).
    RQ2 is fixed in rq2_compositional.py; RQ3's sweep_cfg_scales still uses
    mean_accuracy() and must be re-run after the same fix before its
    accuracy curves are meaningful. CLIPScore and FID are unaffected.
  - Per-set FID at ~50 images is high-variance; prefer pooled FID.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# Consistent colour per pipeline across every figure.
_PIPELINE_COLOURS = {
    "raw_clip":      "#888888",
    "ar_clip":       "#4C9F70",
    "llada_clip":    "#7B5EA7",
    "raw_longclip":  "#BBBBBB",
    "ar_longclip":   "#7FC8A9",
    "llada_longclip":"#A98FD1",
}
_PIPELINE_ORDER = [
    "raw_clip", "ar_clip", "llada_clip",
    "raw_longclip", "ar_longclip", "llada_longclip",
]


def _style():
    """Apply a clean matplotlib style; import here to keep module import light."""
    import matplotlib
    matplotlib.use("Agg")  # headless compute nodes have no display
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
    })
    return plt


def _save(fig, out_dir: Path, name: str, wandb_key: Optional[str] = None) -> None:
    """Write png + pdf and optionally log to W&B."""
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    logger.info("Wrote %s (+ .pdf)", png)

    try:
        import wandb
        if wandb.run is not None:
            wandb.log({wandb_key or f"plots/{name}": wandb.Image(str(png))})
    except Exception as exc:
        logger.debug("W&B image log skipped for %s (%s)", name, exc)

    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Loaders — read what the RQ runners write
# ---------------------------------------------------------------------------

def load_rq2_traces(rq2_dir: Path) -> dict[str, dict[str, list[dict]]]:
    """
    Return {pipeline: {set: [per-prompt records]}} from every trace.jsonl.

    Records carry clip_score, attr_binding{accuracy,n_pairs}, and
    relation_accuracy{accuracy,n_relations}. Missing pipelines/sets are simply
    absent from the returned dict.
    """
    rq2_dir = Path(rq2_dir)
    out: dict[str, dict[str, list[dict]]] = {}
    if not rq2_dir.is_dir():
        logger.warning("RQ2 dir %s not found — no RQ2 plots.", rq2_dir)
        return out
    for trace in sorted(rq2_dir.glob("*/*/trace.jsonl")):
        pipeline = trace.parent.parent.name
        set_name = trace.parent.name
        try:
            records = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines() if l.strip()]
        except Exception as exc:
            logger.warning("Could not read %s (%s)", trace, exc)
            continue
        out.setdefault(pipeline, {})[set_name] = records
    return out


def _applicable_mean(records: list[dict], metric: str) -> Optional[float]:
    """
    Mean of a metric over prompts where it applies (the structural-zero fix).

    metric in {'clip_score', 'attr', 'relation'}.
    Returns None if no applicable prompts (so callers can skip, not plot 0).
    """
    if metric == "clip_score":
        vals = [r["clip_score"] for r in records if "clip_score" in r]
    elif metric == "attr":
        vals = [r["attr_binding"]["accuracy"] for r in records
                if r.get("attr_binding", {}).get("n_pairs", 0) > 0]
    elif metric == "relation":
        vals = [r["relation_accuracy"]["accuracy"] for r in records
                if r.get("relation_accuracy", {}).get("n_relations", 0) > 0]
    else:
        raise ValueError(metric)
    return sum(vals) / len(vals) if vals else None


def load_rq3_sweeps(rq3_dir: Path) -> dict[str, dict]:
    """
    Return {pipeline: {cfg_scales, clip_scores, attr_accuracies, rel_accuracies}}
    from the per-pipeline sidecars written by rq3's _save_summary.

    Reads outputs/rq3/_summary_parts/<pipeline>.json, so it sees ALL pipelines
    regardless of which process wrote them.
    """
    parts = Path(rq3_dir) / "_summary_parts"
    out: dict[str, dict] = {}
    if not parts.is_dir():
        logger.info("No RQ3 sidecars in %s — skipping CFG sweep plots.", parts)
        return out
    for f in sorted(parts.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            out[d["pipeline_name"]] = d
        except Exception as exc:
            logger.warning("Could not read RQ3 sidecar %s (%s)", f, exc)
    return out


def load_rq1_density(rq1_dir: Path) -> dict[str, dict]:
    """
    Return {pipeline: metrics} from RQ1 sidecars if present.

    RQ1 currently writes a summary txt; if a _summary_parts dir exists it is
    used, otherwise this returns empty and RQ1 plots are skipped.
    """
    parts = Path(rq1_dir) / "_summary_parts"
    out: dict[str, dict] = {}
    if not parts.is_dir():
        logger.info("No RQ1 sidecars in %s — skipping RQ1 plots.", parts)
        return out
    for f in sorted(parts.glob("*.json")):
        try:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read RQ1 sidecar %s (%s)", f, exc)
    return out


def plot_rq1_density(rq1_dir: Path, out_dir: Path) -> None:
    """
    RQ1 grouped bars: raw vs rewritten attribute & relation density, and a
    separate separation-gain bar, one group per pipeline.
    """
    plt = _style()
    data = load_rq1_density(rq1_dir)
    if not data:
        return
    import numpy as np

    pipes = [p for p in _PIPELINE_ORDER if p in data]
    if not pipes:
        return

    # Density: raw vs rewritten attribute count.
    fig, ax = plt.subplots(figsize=(1.4 * len(pipes) + 3, 4.5))
    x = np.arange(len(pipes))
    w = 0.35

    def _get(p, key):
        d = data[p]
        # keys may be flat (llada_clip/raw_attr_density) or bare
        for cand in (key, f"{p}/{key}"):
            if cand in d:
                return d[cand]
        return None

    raw_attr = [_get(p, "raw_attr_count") or 0 for p in pipes]
    rw_attr  = [_get(p, "rw_attr_count") or 0 for p in pipes]
    ax.bar(x - w/2, raw_attr, w, label="raw", color="#BBBBBB")
    ax.bar(x + w/2, rw_attr,  w, label="rewritten", color="#7B5EA7")
    ax.set_xticks(x); ax.set_xticklabels(pipes, rotation=20, ha="right")
    ax.set_ylabel("Attribute count (per prompt)")
    ax.set_title("RQ1 — attribute count: raw vs rewritten")
    ax.legend(fontsize=9)
    _save(fig, out_dir, "rq1_attr_count")

    # Separation gain (single bar per pipeline; negative = CLIP clustering).
    sep = [_get(p, "separation_gain") for p in pipes]
    if any(s is not None for s in sep):
        fig, ax = plt.subplots(figsize=(1.2 * len(pipes) + 3, 4.5))
        vals = [s if s is not None else 0 for s in sep]
        colours = ["#C0504D" if v < 0 else "#4C9F70" for v in vals]
        ax.bar(x, vals, color=colours)
        ax.axhline(0, color="#333", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(pipes, rotation=20, ha="right")
        ax.set_ylabel("Separation gain (rw − raw distance)")
        ax.set_title("RQ1 — embedding separation gain (negative = CLIP clustering)")
        _save(fig, out_dir, "rq1_separation_gain")


# ---------------------------------------------------------------------------
# Plot 1: per-set grouped bars, one metric, pipelines side by side
# ---------------------------------------------------------------------------

def plot_rq2_bars(
    rq2_dir: Path,
    out_dir: Path,
    metric: str = "clip_score",
) -> None:
    """
    Grouped bar chart: x = eval set, bars = pipelines, y = metric.

    metric in {'clip_score', 'attr', 'relation'}. Uses the applicable-prompt
    mean, so sets with nothing to score for a metric leave a gap rather than
    a misleading zero.
    """
    plt = _style()
    traces = load_rq2_traces(rq2_dir)
    if not traces:
        return

    pipelines = [p for p in _PIPELINE_ORDER if p in traces]
    all_sets = sorted({s for p in traces.values() for s in p})

    import numpy as np
    x = np.arange(len(all_sets))
    width = 0.8 / max(len(pipelines), 1)

    fig, ax = plt.subplots(figsize=(1.6 * len(all_sets) + 2, 4.5))
    for j, pipe in enumerate(pipelines):
        ys = []
        for s in all_sets:
            recs = traces[pipe].get(s)
            ys.append(_applicable_mean(recs, metric) if recs else None)
        xs = [x[k] + (j - len(pipelines) / 2 + 0.5) * width
              for k, y in enumerate(ys) if y is not None]
        yvals = [y for y in ys if y is not None]
        ax.bar(xs, yvals, width, label=pipe,
               color=_PIPELINE_COLOURS.get(pipe, "#333"))

    label = {"clip_score": "CLIPScore", "attr": "Attribute binding accuracy",
             "relation": "Relation accuracy"}[metric]
    ax.set_xticks(x)
    ax.set_xticklabels(all_sets, rotation=30, ha="right")
    ax.set_ylabel(label)
    ax.set_title(f"RQ2 — {label} by set and pipeline")
    ax.legend(ncol=min(3, len(pipelines)), loc="upper right", fontsize=9)
    _save(fig, out_dir, f"rq2_bars_{metric}")


# ---------------------------------------------------------------------------
# Plot 2: CFG sweep line plots (metric vs. guidance scale)
# ---------------------------------------------------------------------------

def plot_cfg_sweep(
    rq3_results: dict,
    out_dir: Path,
    metric: str = "clip_scores",
) -> None:
    """
    Line plot: x = CFG scale, y = metric, one line per pipeline.

    rq3_results: {pipeline_name: obj|dict} with cfg_scales/clip_scores/
    attr_accuracies/rel_accuracies. Accepts CFGSweepResult objects (live) or
    plain dicts (from load_rq3_sweeps). metric in
    {'clip_scores','attr_accuracies','rel_accuracies'}.
    """
    plt = _style()
    if not rq3_results:
        logger.warning("No RQ3 results passed — skipping CFG sweep plot.")
        return

    import math
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    plotted = 0
    for pipe in _PIPELINE_ORDER:
        r = rq3_results.get(pipe)
        if r is None:
            continue
        scales = getattr(r, "cfg_scales", None)
        if scales is None and isinstance(r, dict):
            scales = r.get("cfg_scales")
        ys = getattr(r, metric, None)
        if ys is None and isinstance(r, dict):
            ys = r.get(metric)
        if not scales or not ys:
            continue
        # Drop NaN points (metrics not applicable to the sweep set).
        pairs = [(s, y) for s, y in zip(scales, ys)
                 if y is not None and not (isinstance(y, float) and math.isnan(y))]
        if not pairs:
            continue
        xs, yv = zip(*pairs)
        ax.plot(xs, yv, marker="o", label=pipe,
                color=_PIPELINE_COLOURS.get(pipe, "#333"))
        plotted += 1

    if plotted == 0:
        logger.warning("CFG sweep: no plottable pipelines for %s.", metric)
        plt.close(fig)
        return

    label = {"clip_scores": "CLIPScore", "attr_accuracies": "Attribute binding",
             "rel_accuracies": "Relation accuracy"}[metric]
    ax.set_xlabel("CFG scale $w$")
    ax.set_ylabel(label)
    ax.set_title(f"RQ3 — {label} vs. guidance scale")
    ax.legend(fontsize=9)
    _save(fig, out_dir, f"rq3_cfg_{metric}")


# ---------------------------------------------------------------------------
# Plot 3: denoising trajectory (metric vs. diffusion timestep)
# ---------------------------------------------------------------------------

def load_trajectory(traj_dir: Path) -> dict[str, dict[str, list[dict]]]:
    """
    Return {pipeline: {set: [ {step, clip_score, ...}, ... ]}} from
    trajectory JSONL files written by generate_with_trajectory().
    """
    traj_dir = Path(traj_dir)
    out: dict[str, dict[str, list[dict]]] = {}
    if not traj_dir.is_dir():
        logger.info("No trajectory dir %s — skipping denoising-step plots. "
                    "Run generation with a scoring callback to produce it.", traj_dir)
        return out
    for f in sorted(traj_dir.glob("*/*/trajectory.jsonl")):
        pipe, set_name = f.parent.parent.name, f.parent.name
        try:
            recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        except Exception as exc:
            logger.warning("Could not read %s (%s)", f, exc)
            continue
        out.setdefault(pipe, {})[set_name] = recs
    return out


def plot_trajectory(
    traj_dir: Path,
    out_dir: Path,
    metric: str = "clip_score",
    set_name: Optional[str] = None,
) -> None:
    """
    Line plot: x = denoising step, y = metric averaged over prompts, one line
    per pipeline. Requires trajectory data (see generate_with_trajectory).
    """
    plt = _style()
    traj = load_trajectory(traj_dir)
    if not traj:
        return

    import numpy as np
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    plotted = 0
    for pipe in _PIPELINE_ORDER:
        if pipe not in traj:
            continue
        sets = [set_name] if set_name else list(traj[pipe])
        # Average metric per step across the chosen set(s)
        by_step: dict[int, list[float]] = {}
        for s in sets:
            for rec in traj[pipe].get(s, []):
                if metric in rec:
                    by_step.setdefault(rec["step"], []).append(rec[metric])
        if not by_step:
            continue
        steps = sorted(by_step)
        ys = [float(np.mean(by_step[s])) for s in steps]
        ax.plot(steps, ys, marker=".", label=pipe,
                color=_PIPELINE_COLOURS.get(pipe, "#333"))
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_xlabel("Denoising step")
    ax.set_ylabel(metric)
    scope = set_name or "all sets"
    ax.set_title(f"Metric trajectory over denoising — {scope}")
    ax.legend(fontsize=9)
    _save(fig, out_dir, f"trajectory_{metric}_{set_name or 'all'}")


# ---------------------------------------------------------------------------
# One-call driver
# ---------------------------------------------------------------------------

def generate_all_plots(
    output_root: str | Path = "outputs",
    rq3_results: Optional[dict] = None,
) -> None:
    """
    Render every available plot from disk artifacts.

    Reads everything from disk (traces + per-pipeline sidecars), so it sees
    ALL pipelines regardless of which process produced them. Safe to call with
    partial data — absent inputs are skipped with a note. rq3_results is
    optional; if not passed, CFG sweeps are loaded from the RQ3 sidecars.
    """
    root = Path(output_root)
    plot_dir = root / "plots"

    # RQ1 — density + separation gain.
    plot_rq1_density(root / "rq1", plot_dir)

    # RQ2 bars — CLIPScore always; accuracy metrics where applicable.
    for m in ("clip_score", "attr", "relation"):
        plot_rq2_bars(root / "rq2", plot_dir, metric=m)

    # RQ3 CFG sweeps — prefer in-memory dict, else load from disk sidecars.
    sweeps = rq3_results or load_rq3_sweeps(root / "rq3")
    if sweeps:
        for m in ("clip_scores", "attr_accuracies", "rel_accuracies"):
            plot_cfg_sweep(sweeps, plot_dir, metric=m)
    else:
        logger.info("No RQ3 data (memory or disk) — CFG sweep plots skipped.")

    # Denoising trajectory — only if instrumented.
    for m in ("clip_score",):
        plot_trajectory(root / "trajectory", plot_dir, metric=m)

    logger.info("Plotting complete. Figures in %s", plot_dir)


if __name__ == "__main__":
    # Standalone: python -m evaluation.plotting [output_root]
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    root = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    generate_all_plots(root)

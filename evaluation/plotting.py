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
        """
        Look up a metric, tolerating (a) pipeline-prefixed vs bare keys and
        (b) the pre-rename *_density names for what are now *_count metrics.
        Old sidecars written before the density→count rename still plot.
        """
        d = data[p]
        # Back-compat aliases: new count name -> possible old density name.
        aliases = {
            "raw_attr_count":  "raw_attr_density",
            "rw_attr_count":   "rw_attr_density",
            "raw_rel_count":   "raw_rel_density",
            "rw_rel_count":    "rw_rel_density",
            "attr_count_gain": "attr_density_gain",
            "rel_count_gain":  "rel_density_gain",
        }
        candidates = [key, f"{p}/{key}"]
        if key in aliases:
            old = aliases[key]
            candidates += [old, f"{p}/{old}"]
        for cand in candidates:
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

    # Semantic density ratio: raw vs rewritten fraction of compositional words.
    # Distinct from the counts above — this is (attrs+rels)/total_words, so it
    # shows whether rewriting CONCENTRATES compositional content or dilutes it
    # with descriptive prose (the RQ1 finding: counts rise but density falls).
    raw_den = [_get(p, "raw_semantic_density") for p in pipes]
    rw_den  = [_get(p, "rw_semantic_density") for p in pipes]
    if any(d is not None for d in raw_den + rw_den):
        fig, ax = plt.subplots(figsize=(1.4 * len(pipes) + 3, 4.5))
        ax.bar(x - w/2, [d or 0 for d in raw_den], w, label="raw", color="#BBBBBB")
        ax.bar(x + w/2, [d or 0 for d in rw_den],  w, label="rewritten", color="#7B5EA7")
        ax.set_xticks(x); ax.set_xticklabels(pipes, rotation=20, ha="right")
        ax.set_ylabel("Semantic density (fraction of words)")
        ax.set_title("RQ1 — semantic density: raw vs rewritten "
                     "(ratio, not count)")
        ax.legend(fontsize=9)
        _save(fig, out_dir, "rq1_semantic_density")


# ---------------------------------------------------------------------------
# Plot 1: per-set grouped bars, one metric, pipelines side by side
# ---------------------------------------------------------------------------

def load_rq2_sidecars(rq2_dir: Path) -> dict[str, dict]:
    """
    Return {pipeline: {set: metrics}} from RQ2 per-pipeline sidecars.

    These carry set-level metrics that the per-prompt traces do NOT — notably
    FID, which is a distribution statistic over the whole image set and so was
    never written per-prompt. Reads outputs/rq2/_summary_parts/<pipeline>.json.
    """
    parts = Path(rq2_dir) / "_summary_parts"
    out: dict[str, dict] = {}
    if not parts.is_dir():
        logger.info("No RQ2 sidecars in %s — skipping FID plot.", parts)
        return out
    for f in sorted(parts.glob("*.json")):
        try:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read RQ2 sidecar %s (%s)", f, exc)
    return out


def plot_rq2_fid(rq2_dir: Path, out_dir: Path) -> None:
    """
    FID grouped bars: x = eval set, bars = pipelines, y = FID (lower better).

    FID lives in the RQ2 sidecars, not the traces, which is why the trace-based
    bar plotter never showed it. Reads the sidecars directly.

    NOTE: per-set FID at ~50 images is high-variance and biased upward; treat
    cross-pipeline differences within a set as indicative, not absolute
    magnitudes. A pooled FID (all sets per pipeline) is more defensible — see
    the horizontal line per pipeline showing its mean FID across sets.
    """
    plt = _style()
    data = load_rq2_sidecars(rq2_dir)
    if not data:
        return
    import numpy as np

    pipelines = [p for p in _PIPELINE_ORDER if p in data]
    all_sets = sorted({s for p in data.values() for s in p})

    # Extract fid per (pipeline, set); key form: '<pipeline>/<set>/fid'.
    def _fid(pipe, s):
        metrics = data.get(pipe, {}).get(s, {})
        return metrics.get(f"{pipe}/{s}/fid")

    # Bail if no FID anywhere (e.g. run without a reference set).
    if not any(_fid(p, s) is not None for p in pipelines for s in all_sets):
        logger.info("RQ2 sidecars contain no FID values — skipping FID plot.")
        return

    x = np.arange(len(all_sets))
    width = 0.8 / max(len(pipelines), 1)
    fig, ax = plt.subplots(figsize=(1.6 * len(all_sets) + 2, 4.5))

    for j, pipe in enumerate(pipelines):
        ys = [_fid(pipe, s) for s in all_sets]
        xs = [x[k] + (j - len(pipelines) / 2 + 0.5) * width
              for k, y in enumerate(ys) if y is not None]
        yvals = [y for y in ys if y is not None]
        ax.bar(xs, yvals, width, label=pipe,
               color=_PIPELINE_COLOURS.get(pipe, "#333"))
        # Pooled mean FID across this pipeline's sets, as a reference line.
        pooled = [y for y in ys if y is not None]
        if pooled:
            ax.axhline(np.mean(pooled), color=_PIPELINE_COLOURS.get(pipe, "#333"),
                       linewidth=0.8, linestyle=":", alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(all_sets, rotation=30, ha="right")
    ax.set_ylabel("FID (lower = better)")
    ax.set_title("RQ2 — FID by set and pipeline (dotted = pipeline mean; "
                 "per-set FID is noisy at small N)")
    ax.legend(ncol=min(3, len(pipelines)), fontsize=9)
    _save(fig, out_dir, "rq2_fid")


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

def plot_rq3_stability(rq3_dir: Path, out_dir: Path) -> None:
    """
    RQ3 compositional stability: one bar per pipeline (higher = less sensitive
    to CFG scale). Reads the compositional_stability field from RQ3 sidecars.

    Stability = 1 - mean(coefficient of variation) across metrics over the CFG
    sweep, so it is comparable across pipelines despite metrics living on
    different scales.
    """
    plt = _style()
    sweeps = load_rq3_sweeps(rq3_dir)
    if not sweeps:
        return
    import numpy as np

    pipes = [p for p in _PIPELINE_ORDER if p in sweeps]
    vals = [sweeps[p].get("compositional_stability") for p in pipes]
    # keep only pipelines with a real (non-NaN) stability value
    pairs = [(p, v) for p, v in zip(pipes, vals)
             if v is not None and not (isinstance(v, float) and v != v)]
    if not pairs:
        logger.info("No RQ3 stability values — skipping stability plot.")
        return
    pipes, vals = zip(*pairs)

    fig, ax = plt.subplots(figsize=(1.3 * len(pipes) + 3, 4.5))
    x = np.arange(len(pipes))
    ax.bar(x, vals, color=[_PIPELINE_COLOURS.get(p, "#333") for p in pipes])
    ax.set_xticks(x); ax.set_xticklabels(pipes, rotation=20, ha="right")
    ax.set_ylabel("Compositional stability (higher = more CFG-robust)")
    ax.set_title("RQ3 — compositional stability across CFG scales")
    ax.set_ylim(top=1.0)
    _save(fig, out_dir, "rq3_stability")


def plot_rq4_deltas(rq4_dir: Path, out_dir: Path) -> None:
    """
    RQ4 mechanism comparison: diverging bars of (LLaDA - AR) per metric per set.

    Reads outputs/rq4/rq4_comparison.json (keys like
    'delta_clip/color_binding/attr_binding_accuracy'). Bars above zero mean
    LLaDA (diffusion) beats AR; below zero means AR wins. This is the core
    RQ4 visual — the mechanism question.
    """
    plt = _style()
    comp_path = Path(rq4_dir) / "rq4_comparison.json"
    if not comp_path.exists():
        logger.info("No %s — skipping RQ4 delta plot.", comp_path)
        return
    try:
        deltas = json.loads(comp_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read RQ4 comparison (%s)", exc)
        return
    if not deltas:
        return

    import numpy as np
    # Parse keys: delta_<encoder>/<set>/<metric>
    parsed = []
    for k, v in deltas.items():
        try:
            enc_part, set_name, metric = k.split("/")
            encoder = enc_part.replace("delta_", "")
        except ValueError:
            continue
        if v != v:  # skip NaN deltas
            continue
        parsed.append((encoder, set_name, metric, v))
    if not parsed:
        return

    metrics = sorted({p[2] for p in parsed})
    metric_labels = {
        "clip_score": "CLIPScore",
        "attr_binding_accuracy": "Attribute binding",
        "relation_accuracy": "Relation accuracy",
    }

    # One subplot per metric; x = set (× encoder), y = delta.
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(4.2 * len(metrics), 4.5), sharey=False
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        rows = [(f"{s}\n({e})", v) for e, s, m, v in parsed if m == metric]
        rows.sort(key=lambda r: r[1])
        labels = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        colours = ["#7B5EA7" if v >= 0 else "#C0504D" for v in vals]
        y = np.arange(len(labels))
        ax.barh(y, vals, color=colours)
        ax.axvline(0, color="#333", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("LLaDA - AR")
        ax.set_title(metric_labels.get(metric, metric), fontsize=10)

    fig.suptitle("RQ4 - Mechanism deltas (>0: diffusion beats autoregressive)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir, "rq4_mechanism_deltas")


def plot_rewrite_timing(output_root: Path, out_dir: Path) -> None:
    """
    Language-layer inference time: AR vs LLaDA wall-clock per rewrite.

    Reads outputs/rewrite_timing.json (written by the RQ0 warmup). Two bars
    per mechanism: mean and median seconds per rewrite. This is the concrete
    mechanism-cost tradeoff — if LLaDA costs Nx the AR wall-clock for
    comparable quality, that N is a first-class finding.
    """
    plt = _style()
    timing_path = Path(output_root) / "rewrite_timing.json"
    if not timing_path.exists():
        logger.info("No %s — skipping rewrite-timing plot.", timing_path)
        return
    try:
        data = json.loads(timing_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read rewrite timing (%s)", exc)
        return
    if not data:
        return

    import numpy as np
    # data: {pipeline_name: {mechanism, n, mean_seconds, median_seconds, ...}}
    names = sorted(data.keys())
    means = [data[n].get("mean_seconds", 0) for n in names]
    medians = [data[n].get("median_seconds", 0) for n in names]
    counts = [data[n].get("n", 0) for n in names]

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 3, 4.5))
    ax.bar(x - w/2, means, w, label="mean", color="#7B5EA7")
    ax.bar(x + w/2, medians, w, label="median", color="#4C9F70")

    # Annotate with per-mechanism sample count and the LLaDA/AR ratio if both present.
    for xi, (m, c) in enumerate(zip(means, counts)):
        ax.annotate(f"n={c}", (xi, m), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Seconds per rewrite")
    ax.set_title("Language-layer inference time (AR vs LLaDA)")
    ax.legend(fontsize=9)

    # If both an ar_* and llada_* pipeline are present, print the cost ratio.
    ar = next((n for n in names if n.startswith("ar_")), None)
    ll = next((n for n in names if n.startswith("llada_")), None)
    if ar and ll and data[ar].get("mean_seconds"):
        ratio = data[ll]["mean_seconds"] / data[ar]["mean_seconds"]
        ax.text(0.98, 0.02, f"LLaDA / AR mean = {ratio:.1f}x",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color="#333",
                bbox=dict(boxstyle="round", fc="#f4f0fa", ec="#7B5EA7", alpha=0.8))
    _save(fig, out_dir, "rewrite_timing")


def load_cfg_stability(stab_dir: Path) -> dict[str, dict]:
    """
    Return {pipeline: {cfg_scale(float): stability}} from cfg_stability runs.
    Reads outputs/cfg_stability/<pipeline>/stability.json.
    """
    stab_dir = Path(stab_dir)
    out: dict[str, dict] = {}
    if not stab_dir.is_dir():
        logger.info("No cfg_stability dir %s — skipping stability-vs-CFG plots.", stab_dir)
        return out
    for f in sorted(stab_dir.glob("*/stability.json")):
        pipe = f.parent.name
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            out[pipe] = {float(k): v for k, v in raw.items()}
        except Exception as exc:
            logger.warning("Could not read %s (%s)", f, exc)
    return out


def plot_cfg_stability(stab_dir: Path, out_dir: Path) -> None:
    """
    Trajectory stability vs CFG scale, one line per pipeline (higher = the
    denoising path stays smoother at that guidance strength).

    Distinct from RQ3's compositional_stability (variance of FINAL quality
    across scales): this measures step-to-step smoothness of the denoising
    TRAJECTORY at each scale. NOTE: the underlying 1/(1+mean|Δ|) metric is an
    ad-hoc smoothness proxy — read differences relatively, not absolutely.
    """
    plt = _style()
    data = load_cfg_stability(stab_dir)
    if not data:
        return
    import math

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    plotted = 0
    for pipe in _PIPELINE_ORDER:
        if pipe not in data:
            continue
        items = sorted(data[pipe].items())
        pairs = [(s, v) for s, v in items
                 if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        ax.plot(xs, ys, marker="o", label=pipe,
                color=_PIPELINE_COLOURS.get(pipe, "#333"))
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("CFG scale $w$")
    ax.set_ylabel("Trajectory stability (higher = smoother path)")
    ax.set_title("Denoising-trajectory stability vs. guidance scale")
    ax.legend(fontsize=9)
    _save(fig, out_dir, "cfg_stability")


def plot_tunability_heatmaps(rq6_dir: Path, out_dir: Path) -> None:
    """
    RQ6 tunability: heatmaps over the 16x16 (LLaDA x image) config grid for
    quality (CLIPScore), total time, and generation time. Reads
    outputs/<backbone>/rq6/grid.jsonl.

    Axes: rows = LLaDA config index (steps x gen_length),
          cols = image config index (num_inference_steps x cfg_scale).
    The SHAPE of the surface is the result — which region is fast/good, and
    how steerable each axis is (steepness of the gradient).
    """
    plt = _style()
    grid_path = Path(rq6_dir) / "grid.jsonl"
    if not grid_path.exists():
        logger.info("No %s — skipping tunability heatmaps.", grid_path)
        return
    import numpy as np

    rows = []
    for line in grid_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        return

    n_l = max(r["llada_idx"] for r in rows) + 1
    n_i = max(r["image_idx"] for r in rows) + 1

    def _grid(key):
        g = np.full((n_l, n_i), np.nan)
        for r in rows:
            g[r["llada_idx"], r["image_idx"]] = r.get(key, np.nan)
        return g

    panels = [
        ("mean_clip_score", "CLIPScore (quality)", "viridis"),
        ("total_time", "Total time (s/prompt)", "magma_r"),
        ("mean_gen_time", "Image gen time (s/prompt)", "magma_r"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.6))
    if len(panels) == 1:
        axes = [axes]

    for ax, (key, title, cmap) in zip(axes, panels):
        g = _grid(key)
        im = ax.imshow(g, aspect="auto", cmap=cmap, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("image config idx\n(steps x cfg)")
        ax.set_ylabel("LLaDA config idx\n(steps x gen_length)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate with steerability if available.
    stab_path = Path(rq6_dir) / "steerability.json"
    subtitle = ""
    if stab_path.exists():
        try:
            s = json.loads(stab_path.read_text(encoding="utf-8"))
            subtitle = (f"steerability: image-axis={s['image_axis_steerability']:.4f}, "
                        f"LLaDA-axis={s['llada_axis_steerability']:.4f} "
                        f"(mean |ΔCLIP| between adjacent cells)")
        except Exception:
            pass
    fig.suptitle("RQ6 — tunability surface" + (f"\n{subtitle}" if subtitle else ""),
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, out_dir, "rq6_tunability")


def plot_tunability_image_grid(rq6_dir: Path, out_dir: Path,
                              llada_vary: str = "steps",
                              image_vary: str = "cfg_scale") -> None:
    """
    RQ6 tunability as an IMAGE grid (4x4): actually shows the generated images
    across the two tuning axes, so you can SEE how the output changes as you
    turn each knob — not just a metric heatmap.

    rows = 4 values of the LLaDA knob `llada_vary` (steps | gen_length)
    cols = 4 values of the image knob `image_vary` (cfg_scale | num_inference_steps)
    the OTHER knob on each layer is held fixed (at its first value) so each axis
    is one interpretable dimension.

    Reads the showcase images saved per cell at rq6/grid_images/L{li}_I{ii}.png.
    Cell indexing (from _build_axes): li = steps_idx*4 + genlen_idx,
    ii = imgsteps_idx*4 + cfg_idx, with values {32,64,96,128} / {10,25,50,100} /
    cfg {2,5,7.5,12}.
    """
    plt = _style()
    img_dir = Path(rq6_dir) / "grid_images"
    if not img_dir.is_dir() or not any(img_dir.glob("*.png")):
        logger.info("No rq6 showcase images in %s — skipping tunability image grid.", img_dir)
        return
    from PIL import Image

    llada_steps = [32, 64, 96, 128]
    llada_genlen = [32, 64, 96, 128]
    img_steps = [10, 25, 50, 100]
    img_cfg = [2.0, 5.0, 7.5, 12.0]

    # Build the 4 LLaDA row indices and 4 image col indices for the chosen
    # varying knob, holding the other at index 0.
    if llada_vary == "steps":
        row_vals = llada_steps
        li_of = lambda r: r * 4 + 0            # vary steps, gen_length fixed at idx0
        row_label = "LLaDA steps"
    else:
        row_vals = llada_genlen
        li_of = lambda r: 0 * 4 + r            # vary gen_length, steps fixed at idx0
        row_label = "LLaDA gen_length"

    if image_vary == "cfg_scale":
        col_vals = img_cfg
        ii_of = lambda c: 0 * 4 + c            # vary cfg, img_steps fixed at idx0
        col_label = "image CFG scale"
    else:
        col_vals = img_steps
        ii_of = lambda c: c * 4 + 0            # vary img_steps, cfg fixed at idx0
        col_label = "image inference steps"

    fig, axes = plt.subplots(4, 4, figsize=(11, 11.6), squeeze=False)
    for r in range(4):
        for c in range(4):
            ax = axes[r][c]
            ax.axis("off")
            p = img_dir / f"L{li_of(r)}_I{ii_of(c)}.png"
            if p.exists():
                try:
                    ax.imshow(Image.open(p).convert("RGB"))
                except Exception:
                    ax.text(0.5, 0.5, "load err", ha="center", va="center", fontsize=8)
            else:
                ax.text(0.5, 0.5, "(missing)", ha="center", va="center",
                        fontsize=9, color="#999", transform=ax.transAxes)
            if r == 0:
                ax.set_title(f"{col_label.split()[-1]}={col_vals[c]:g}", fontsize=9)
            if c == 0:
                ax.annotate(f"{row_vals[r]:g}", xy=(0, 0.5), xytext=(-8, 0),
                            textcoords="offset points", xycoords="axes fraction",
                            ha="right", va="center", fontsize=9, fontweight="bold")

    fig.suptitle(f"RQ6 tunability — generated images across tuning axes\n"
                 f"rows: {row_label} (top→bottom)   cols: {col_label} (left→right)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0.03, 0, 1, 0.95])
    _save(fig, out_dir, "rq6_image_grid")


def plot_trajectory_cfg_lines(stab_dir: Path, out_dir: Path) -> None:
    """
    Denoising trajectory as LINE GRAPHS: metric (CLIPScore) vs denoising STEP,
    one line per CFG scale, per pipeline. Shows how quality evolves THROUGH
    generation and how guidance strength changes that evolution.

    Reads the per-step records cfg_stability writes at
    outputs/<backbone>/cfg_stability/<pipeline>/trajectory_cfg.jsonl
    (fields: idx, cfg, step, clip_score), averaging over prompts per (cfg, step).
    """
    plt = _style()
    stab_dir = Path(stab_dir)
    if not stab_dir.is_dir():
        logger.info("No cfg_stability dir %s — skipping trajectory line graphs.", stab_dir)
        return
    import numpy as np
    from collections import defaultdict

    traj_files = sorted(stab_dir.glob("*/trajectory_cfg.jsonl"))
    if not traj_files:
        logger.info("No trajectory_cfg.jsonl under %s — skipping.", stab_dir)
        return

    for tf in traj_files:
        pipe = tf.parent.name
        # (cfg, step) -> list of clip_scores
        agg: dict = defaultdict(list)
        for line in tf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                agg[(float(r["cfg"]), int(r["step"]))].append(r["clip_score"])
            except Exception:
                continue
        if not agg:
            continue

        cfgs = sorted({k[0] for k in agg})
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        cmap = plt.cm.viridis(np.linspace(0, 0.9, len(cfgs)))
        for col, cfg_v in zip(cmap, cfgs):
            pts = sorted((s, np.mean(agg[(cfg_v, s)]))
                         for (cv, s) in agg if cv == cfg_v)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", ms=3, color=col, label=f"CFG {cfg_v:g}")
        ax.set_xlabel("denoising step")
        ax.set_ylabel("CLIPScore")
        ax.set_title(f"Denoising trajectory by guidance scale — {pipe}")
        ax.legend(fontsize=8, title="guidance")
        _save(fig, out_dir, f"trajectory_cfg_lines_{pipe}")


def plot_llada_trajectory(traj_dir: Path, out_dir: Path) -> None:
    """
    Language-layer trajectory: how a LLaDA rewrite emerges across unmasking
    steps. Two panels:
      (left)  unmasked fraction vs step — the unmasking schedule (mean +
              individual prompt lines, faint).
      (right) compositional content vs step — semantic density of the partial
              decode (only if --track-content was used; skipped otherwise).

    Reads outputs/<backbone>/llada_trajectory/trajectory.jsonl.
    Visualises the diffusion mechanism's joint resolution — the language-side
    counterpart to the image denoising trajectory, and unique vs AR.
    """
    plt = _style()
    traj_path = Path(traj_dir) / "trajectory.jsonl"
    if not traj_path.exists():
        logger.info("No %s — skipping LLaDA trajectory plot.", traj_path)
        return
    import numpy as np
    from collections import defaultdict

    rows = []
    for line in traj_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return

    has_content = any("semantic_density" in r for r in rows)
    n_panels = 2 if has_content else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.0 * n_panels, 4.5), squeeze=False)
    axes = axes[0]

    # Panel 1: unmasked fraction vs step.
    by_idx = defaultdict(list)
    for r in rows:
        by_idx[r["idx"]].append((r["step"], r.get("unmasked_fraction", np.nan)))
    # faint per-prompt lines + bold mean
    step_to_vals = defaultdict(list)
    for idx, pts in by_idx.items():
        pts.sort()
        xs, ys = zip(*pts)
        axes[0].plot(xs, ys, color="#7B5EA7", alpha=0.15, linewidth=0.8)
        for s, v in pts:
            step_to_vals[s].append(v)
    steps = sorted(step_to_vals)
    mean_unmask = [np.nanmean(step_to_vals[s]) for s in steps]
    axes[0].plot(steps, mean_unmask, color="#7B5EA7", linewidth=2.4, label="mean")
    axes[0].set_xlabel("unmasking step")
    axes[0].set_ylabel("fraction of response unmasked")
    axes[0].set_title("LLaDA unmasking schedule")
    axes[0].legend(fontsize=9)

    # Panel 2: semantic density vs step (compositional content emerging).
    if has_content:
        dens = defaultdict(list)
        for r in rows:
            if "semantic_density" in r:
                dens[r["step"]].append(r["semantic_density"])
        dsteps = sorted(dens)
        dmean = [np.nanmean(dens[s]) for s in dsteps]
        axes[1].plot(dsteps, dmean, color="#4C9F70", linewidth=2.4, marker="o", ms=3)
        axes[1].set_xlabel("unmasking step")
        axes[1].set_ylabel("semantic density of partial decode")
        axes[1].set_title("Compositional content emerging through unmasking")

    fig.suptitle("LLaDA language-layer trajectory (joint resolution)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir, "llada_trajectory")


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

    # RQ2 FID — reads sidecars (traces don't carry FID).
    plot_rq2_fid(root / "rq2", plot_dir)

    # RQ4 mechanism deltas (LLaDA vs AR) — the core comparison plot.
    plot_rq4_deltas(root / "rq4", plot_dir)

    # RQ3 CFG sweeps — prefer in-memory dict, else load from disk sidecars.
    sweeps = rq3_results or load_rq3_sweeps(root / "rq3")
    if sweeps:
        for m in ("clip_scores", "attr_accuracies", "rel_accuracies"):
            plot_cfg_sweep(sweeps, plot_dir, metric=m)
    else:
        logger.info("No RQ3 data (memory or disk) — CFG sweep plots skipped.")

    # RQ3 compositional stability bars.
    plot_rq3_stability(root / "rq3", plot_dir)

    # Denoising trajectory — only if instrumented.
    for m in ("clip_score",):
        plot_trajectory(root / "trajectory", plot_dir, metric=m)

    # Language-layer inference timing (AR vs LLaDA wall-clock).
    plot_rewrite_timing(root, plot_dir)

    # CFG-stability across denoising steps (diagnostic).
    plot_cfg_stability(root / "cfg_stability", plot_dir)

    # RQ6 tunability heatmaps (if the sweep ran).
    plot_tunability_heatmaps(root / "rq6", plot_dir)
    # RQ6 tunability IMAGE grid (the visual figure).
    plot_tunability_image_grid(root / "rq6", plot_dir)
    # Multi-CFG denoising trajectory line graphs (if cfg_stability ran).
    plot_trajectory_cfg_lines(root / "cfg_stability", plot_dir)
    # LLaDA language-layer trajectory (if the diagnostic ran).
    plot_llada_trajectory(root / "llada_trajectory", plot_dir)

    logger.info("Plotting complete. Figures in %s", plot_dir)


if __name__ == "__main__":
    # Standalone: python -m evaluation.plotting [output_root]
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    root = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    generate_all_plots(root)

"""
experiments/rq3_cfg_sensitivity.py

RQ3: Does prompt refinement reduce sensitivity to CFG scale?

Sweeps CFG scales {1, 3, 5, 7.5, 10} for each pipeline × encoder using
the rq3_sweep prompt set. Measures variance in CLIPScore, attribute binding,
and relation accuracy across scales as a proxy for compositional stability.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_rq3(
    pipelines: list,
    runner,
    prompts: list[str],
    cfg_scales: list[float],
    clip_scorer,
    attr_scorer,
    rel_scorer,
    seed: int = 42,
    output_dir: str | Path = "outputs/rq3",
    wandb_log: bool = True,
) -> dict:
    """
    Run CFG sensitivity sweep for all pipelines.

    Parameters
    ----------
    pipelines : list[ConditioningPipeline]
    runner : T2IRunner
    prompts : list[str]
        Prompts from the rq3_sweep set in prompts.yaml.
    cfg_scales : list[float]
        e.g. [1.0, 3.0, 5.0, 7.5, 10.0]
    clip_scorer, attr_scorer, rel_scorer : scorers
    seed : int
        Fixed seed — CFG scale is the only variable.
    output_dir : str | Path
    wandb_log : bool

    Returns
    -------
    dict mapping pipeline_name → CFGSweepResult
    """
    from evaluation.cfg_sensitivity import sweep_cfg_scales, compare_stability
    from utils.logging import log_metrics

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_results = []

    for pipeline in pipelines:
        logger.info("[RQ3] Running CFG sweep for: %s", pipeline.name)
        result = sweep_cfg_scales(
            pipeline=pipeline,
            runner=runner,
            prompts=prompts,
            cfg_scales=cfg_scales,
            clip_scorer=clip_scorer,
            attr_scorer=attr_scorer,
            rel_scorer=rel_scorer,
            seed=seed,
            output_dir=str(output_dir),
        )
        sweep_results.append(result)

        if wandb_log:
            log_metrics(result.to_dict())

    # Cross-pipeline stability comparison
    stability_metrics = compare_stability(sweep_results)
    if wandb_log:
        log_metrics(stability_metrics)

    _save_summary(sweep_results, output_dir / "rq3_summary.txt")
    return {r.pipeline_name: r for r in sweep_results}


def _save_summary(results, path: Path) -> None:
    """
    Persist per-pipeline CFG-sweep results and compose a combined summary.

    Each pipeline runs as a separate process, so results go to per-pipeline
    JSON sidecars and the .txt is rebuilt from all sidecars — avoids the
    last-writer-clobbers-all bug where only llada_clip survived.
    """
    path = Path(path)
    parts_dir = path.parent / "_summary_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    def _fmt(v: float) -> str:
        return f"{v:.4f}" if v == v else "n/a"

    # 1. Dump this process's pipeline(s) to sidecars.
    for r in results:
        payload = {
            "pipeline_name": r.pipeline_name,
            "compositional_stability": r.compositional_stability,
            "clip_score_variance": r.clip_score_variance,
            "attr_accuracy_variance": r.attr_accuracy_variance,
            "rel_accuracy_variance": r.rel_accuracy_variance,
            "cfg_scales": list(r.cfg_scales),
            "clip_scores": list(r.clip_scores),
            "attr_accuracies": list(r.attr_accuracies),
            "rel_accuracies": list(r.rel_accuracies),
        }
        with open(parts_dir / f"{r.pipeline_name}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    # 2. Load every sidecar.
    merged = []
    for part in sorted(parts_dir.glob("*.json")):
        try:
            with open(part, encoding="utf-8") as f:
                merged.append(json.load(f))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping unreadable RQ3 part %s (%s)", part, exc)

    # 3. Rank (nan stability last) and write combined summary.
    merged.sort(key=lambda d: (
        d["compositional_stability"] != d["compositional_stability"],
        -d["compositional_stability"] if d["compositional_stability"] == d["compositional_stability"] else 0,
    ))
    with open(path, "w", encoding="utf-8") as f:
        f.write("RQ3 - CFG Sensitivity Results\n")
        f.write("=" * 50 + "\n\n")
        for d in merged:
            f.write(f"Pipeline: {d['pipeline_name']}\n")
            f.write(f"  Compositional stability: {_fmt(d['compositional_stability'])}\n")
            f.write(f"  CLIPScore variance:      {d['clip_score_variance']:.6f}\n")
            f.write(f"  Attr accuracy variance:  {d['attr_accuracy_variance']:.6f}\n")
            f.write(f"  Rel accuracy variance:   {d['rel_accuracy_variance']:.6f}\n")
            for scale, clip, attr, rel in zip(
                d["cfg_scales"], d["clip_scores"], d["attr_accuracies"], d["rel_accuracies"]
            ):
                f.write(f"    CFG={scale}: CLIP={_fmt(clip)} "
                        f"Attr={_fmt(attr)} Rel={_fmt(rel)}\n")
            f.write("\n")
    logger.info("RQ3 summary saved to %s (%d pipelines)", path, len(merged))

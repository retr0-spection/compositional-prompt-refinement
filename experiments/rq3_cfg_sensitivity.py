"""
experiments/rq3_cfg_sensitivity.py

RQ3: Does prompt refinement reduce sensitivity to CFG scale?

Sweeps CFG scales {1, 3, 5, 7.5, 10} for each pipeline × encoder using
the rq3_sweep prompt set. Measures variance in CLIPScore, attribute binding,
and relation accuracy across scales as a proxy for compositional stability.
"""

from __future__ import annotations

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
    with open(path, "w") as f:
        f.write("RQ3 — CFG Sensitivity Results\n")
        f.write("=" * 50 + "\n\n")
        ranked = sorted(results, key=lambda r: r.compositional_stability, reverse=True)
        for r in ranked:
            f.write(f"Pipeline: {r.pipeline_name}\n")
            f.write(f"  Compositional stability: {r.compositional_stability:.4f}\n")
            f.write(f"  CLIPScore variance:      {r.clip_score_variance:.6f}\n")
            f.write(f"  Attr accuracy variance:  {r.attr_accuracy_variance:.6f}\n")
            f.write(f"  Rel accuracy variance:   {r.rel_accuracy_variance:.6f}\n")
            for scale, clip, attr, rel in zip(
                r.cfg_scales, r.clip_scores, r.attr_accuracies, r.rel_accuracies
            ):
                f.write(f"    CFG={scale}: CLIP={clip:.4f} Attr={attr:.4f} Rel={rel:.4f}\n")
            f.write("\n")
    logger.info("RQ3 summary saved to %s", path)

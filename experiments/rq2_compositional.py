"""
experiments/rq2_compositional.py

RQ2: Does improved conditioning reduce compositional errors?

For each pipeline × encoder × seed:
  1. Encode all prompts in the eval sets
  2. Generate images (SD 2.1, default CFG scale)
  3. Score with CLIPScore, attribute binding accuracy, relation accuracy
  4. Compute FID against a reference set (if provided)

Results are logged to W&B and saved to output_dir.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


def run_rq2(
    pipelines: list,
    runner,                         # T2IRunner
    prompt_sets: dict[str, list[str]],  # {set_name: [prompts]}
    clip_scorer,
    attr_scorer,
    rel_scorer,
    fid_scorer=None,
    seed: int = 42,
    cfg_scale: float = 7.5,
    output_dir: str | Path = "outputs/rq2",
    wandb_log: bool = True,
    reference_images: Optional[list[Image.Image]] = None,
) -> dict[str, dict]:
    """
    Run RQ2 compositional benchmark for all pipelines.

    Parameters
    ----------
    pipelines : list[ConditioningPipeline]
    runner : T2IRunner
    prompt_sets : dict[str, list[str]]
        Dict of {set_name: prompts}. Keys should match eval_prompt_sets.rq2
        in experiment.yaml.
    clip_scorer : CLIPScorer
    attr_scorer : AttributeBindingScorer
    rel_scorer : RelationAccuracyScorer
    fid_scorer : FIDScorer | None
        If provided, FID is computed against reference_images.
    seed : int
    cfg_scale : float
    output_dir : str | Path
    wandb_log : bool
    reference_images : list[PIL.Image] | None
        Reference images for FID. If None, FID is skipped.

    Returns
    -------
    dict mapping pipeline_name → {set_name: metric_dict}
    """
    from evaluation.metrics import EvalResult, score_all
    from utils.logging import log_metrics
    from utils.seed import get_generator

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    # FID reference strategy:
    # If reference_images are explicitly provided, use them for all pipelines.
    # Otherwise, on the first pipeline pass (typically raw_clip), we collect
    # its generated images as the FID reference distribution for all subsequent
    # pipelines. This gives us relative FID (vs raw) rather than absolute FID
    # (vs COCO), which is still a valid ablation signal.
    _fid_reference: Optional[list[Image.Image]] = reference_images
    _raw_pipeline_images: list[Image.Image] = []

    if fid_scorer and _fid_reference:
        logger.info("Loading %d explicit reference images into FID scorer", len(_fid_reference))
        fid_scorer.update_real(_fid_reference)

    for pipeline in pipelines:
        logger.info("[RQ2] Pipeline: %s | CFG=%.1f | Seed=%d", pipeline.name, cfg_scale, seed)
        pipeline_results: dict[str, dict] = {}

        for set_name, prompts in prompt_sets.items():
            logger.info("[RQ2][%s] Encoding %d prompts from '%s'",
                        pipeline.name, len(prompts), set_name)

            encoding_results = pipeline.encode_batch(prompts)
            embeddings = [r.embedding for r in encoding_results]
            seeds = [seed] * len(prompts)

            logger.info("[RQ2][%s] Generating images...", pipeline.name)
            images = runner.generate_batch(
                prompt_embeds_list=embeddings,
                cfg_scale=cfg_scale,
                seeds=seeds,
            )

            # Save images
            img_dir = output_dir / pipeline.name / set_name
            img_dir.mkdir(parents=True, exist_ok=True)
            for i, (img, prompt) in enumerate(zip(images, prompts)):
                img.save(img_dir / f"prompt_{i:03d}.png")

            # Collect raw pipeline images as FID reference (first pipeline only)
            if fid_scorer and _fid_reference is None:
                _raw_pipeline_images.extend(images)

            # Score
            result = score_all(
                pipeline_name=pipeline.name,
                images=images,
                prompts=prompts,
                clip_scorer=clip_scorer,
                attr_scorer=attr_scorer,
                rel_scorer=rel_scorer,
            )

            # FID — only score non-raw pipelines (comparing against raw baseline)
            if fid_scorer and _fid_reference is not None:
                fid_scorer.update_generated(images)
                result.fid = fid_scorer.compute()
                fid_scorer.reset()
                fid_scorer.update_real(_fid_reference)

            metrics = {
                f"{pipeline.name}/{set_name}/clip_score": result.clip_score,
                f"{pipeline.name}/{set_name}/attr_binding_accuracy": result.attr_binding_accuracy,
                f"{pipeline.name}/{set_name}/relation_accuracy": result.relation_accuracy,
            }
            if result.fid is not None:
                metrics[f"{pipeline.name}/{set_name}/fid"] = result.fid

            pipeline_results[set_name] = metrics

            if wandb_log:
                log_metrics(metrics)

            logger.info(
                "[RQ2][%s][%s] CLIP=%.4f | Attr=%.4f | Rel=%.4f | FID=%s",
                pipeline.name, set_name,
                result.clip_score, result.attr_binding_accuracy, result.relation_accuracy,
                f"{result.fid:.2f}" if result.fid else "N/A",
            )

        all_results[pipeline.name] = pipeline_results

        # After the first pipeline completes, promote its images to FID reference
        # so all subsequent pipelines are scored relative to raw output.
        if fid_scorer and _fid_reference is None and _raw_pipeline_images:
            _fid_reference = list(_raw_pipeline_images)
            _raw_pipeline_images.clear()
            logger.info(
                "Using %d images from '%s' as FID reference for remaining pipelines.",
                len(_fid_reference), pipeline.name,
            )
            fid_scorer.update_real(_fid_reference)

    _save_summary(all_results, output_dir / "rq2_summary.txt")
    return all_results


def _save_summary(results: dict, path: Path) -> None:
    with open(path, "w") as f:
        f.write("RQ2 — Compositional Benchmark Results\n")
        f.write("=" * 50 + "\n\n")
        for pipeline_name, set_results in results.items():
            f.write(f"Pipeline: {pipeline_name}\n")
            for set_name, metrics in set_results.items():
                f.write(f"  Set: {set_name}\n")
                for k, v in sorted(metrics.items()):
                    f.write(f"    {k}: {v:.4f}\n")
            f.write("\n")
    logger.info("RQ2 summary saved to %s", path)

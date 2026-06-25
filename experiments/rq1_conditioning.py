"""
experiments/rq1_conditioning.py

RQ1: Does diffusion-based refinement produce more semantically structured
conditioning signals than raw prompts?

Analysis:
  1. Semantic density comparison (attribute/relation token counts)
     between raw and rewritten prompts.
  2. CLIP embedding cosine separation — do rewritten prompts produce
     more distinct conditioning vectors across the prompt set?

No image generation required for this RQ — it operates on text and embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def run_rq1(
    pipelines: list,          # list[ConditioningPipeline]
    prompts: list[str],
    output_dir: str | Path = "outputs/rq1",
    wandb_log: bool = True,
) -> dict[str, dict]:
    """
    Run RQ1 analysis for all pipelines.

    Parameters
    ----------
    pipelines : list[ConditioningPipeline]
        All six pipeline × encoder combinations (or a subset for dry runs).
    prompts : list[str]
        Raw prompts (from color_binding, shape_binding, texture_binding, spatial_relations sets).
    output_dir : str | Path
    wandb_log : bool

    Returns
    -------
    dict mapping pipeline_name → {semantic_density_stats, embedding_separation_stats}
    """
    from evaluation.embedding_analysis import (
        analyse_semantic_density,
        compare_embedding_separation,
    )
    from utils.logging import log_metrics

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for pipeline in pipelines:
        logger.info("[RQ1] Processing pipeline: %s", pipeline.name)

        # Encode all prompts — captures both raw and rewritten text
        encoding_results = pipeline.encode_batch(prompts)

        raw_prompts = [r.raw_prompt for r in encoding_results]
        rewritten_prompts = [r.rewritten_prompt for r in encoding_results]
        raw_embeddings = torch.cat([r.embedding for r in encoding_results], dim=0)

        # Also encode raw prompts through the same encoder (for fair comparison)
        # For RawPipeline, raw == rewritten; for others, encode raw separately
        from pipeline.raw import RawPipeline
        raw_only_results = []
        if not isinstance(pipeline, RawPipeline):
            # Use the same encoder as this pipeline but without rewriting
            raw_pipeline = RawPipeline(encoder=pipeline._encoder)
            raw_only_results = raw_pipeline.encode_batch(prompts)
            raw_only_embeddings = torch.cat(
                [r.embedding for r in raw_only_results], dim=0
            )
        else:
            raw_only_embeddings = raw_embeddings

        # 1. Semantic density analysis
        density_stats = analyse_semantic_density(
            raw_prompts=raw_prompts,
            rewritten_prompts=rewritten_prompts,
            pipeline_name=pipeline.name,
        )

        # 2. Embedding separation analysis
        separation_stats = compare_embedding_separation(
            raw_embeddings=raw_only_embeddings,
            rewritten_embeddings=raw_embeddings,
            pipeline_name=pipeline.name,
        )

        combined = {**density_stats, **separation_stats}
        results[pipeline.name] = combined

        if wandb_log:
            log_metrics(combined)

        logger.info(
            "[RQ1][%s] Semantic density gain: attr=%.2f, rel=%.2f | "
            "Separation gain: %.4f",
            pipeline.name,
            density_stats.get(f"{pipeline.name}/attr_density_gain", 0),
            density_stats.get(f"{pipeline.name}/rel_density_gain", 0),
            separation_stats.get(f"{pipeline.name}/separation_gain", 0),
        )

    # Save summary
    _save_summary(results, output_dir / "rq1_summary.txt")
    return results


def _save_summary(results: dict, path: Path) -> None:
    with open(path, "w") as f:
        f.write("RQ1 — Conditioning Structure Analysis\n")
        f.write("=" * 50 + "\n\n")
        for pipeline_name, stats in results.items():
            f.write(f"Pipeline: {pipeline_name}\n")
            for k, v in sorted(stats.items()):
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")
    logger.info("RQ1 summary saved to %s", path)

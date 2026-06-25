"""
evaluation/cfg_sensitivity.py

RQ3: Does prompt refinement reduce sensitivity to classifier-free guidance scale?

Runs each pipeline across CFG scales w ∈ {1, 3, 5, 7.5, 10} and measures
how much metric variance each pipeline exhibits across scales. Lower variance
= more compositionally stable conditioning.

Key function: sweep_cfg_scales()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class CFGSweepResult:
    """Results for one pipeline across all CFG scales."""
    pipeline_name: str
    cfg_scales: list[float]

    # Per-scale aggregated metrics (one value per cfg_scale)
    clip_scores: list[float] = field(default_factory=list)
    attr_accuracies: list[float] = field(default_factory=list)
    rel_accuracies: list[float] = field(default_factory=list)

    @property
    def clip_score_variance(self) -> float:
        return float(np.var(self.clip_scores)) if self.clip_scores else 0.0

    @property
    def attr_accuracy_variance(self) -> float:
        return float(np.var(self.attr_accuracies)) if self.attr_accuracies else 0.0

    @property
    def rel_accuracy_variance(self) -> float:
        return float(np.var(self.rel_accuracies)) if self.rel_accuracies else 0.0

    @property
    def compositional_stability(self) -> float:
        """
        Composite stability score = 1 - normalised average variance.

        Higher is better (less sensitive to CFG scale).
        Normalised over the expected range of each metric.
        """
        return 1.0 - (
            self.clip_score_variance
            + self.attr_accuracy_variance
            + self.rel_accuracy_variance
        ) / 3.0

    def to_dict(self) -> dict:
        """Flat dict for W&B logging."""
        out = {
            f"{self.pipeline_name}/cfg_clip_variance": self.clip_score_variance,
            f"{self.pipeline_name}/cfg_attr_variance": self.attr_accuracy_variance,
            f"{self.pipeline_name}/cfg_rel_variance": self.rel_accuracy_variance,
            f"{self.pipeline_name}/compositional_stability": self.compositional_stability,
        }
        for i, scale in enumerate(self.cfg_scales):
            tag = f"cfg{scale}"
            if i < len(self.clip_scores):
                out[f"{self.pipeline_name}/{tag}/clip_score"] = self.clip_scores[i]
            if i < len(self.attr_accuracies):
                out[f"{self.pipeline_name}/{tag}/attr_accuracy"] = self.attr_accuracies[i]
            if i < len(self.rel_accuracies):
                out[f"{self.pipeline_name}/{tag}/rel_accuracy"] = self.rel_accuracies[i]
        return out


def sweep_cfg_scales(
    pipeline,                        # ConditioningPipeline
    runner,                          # T2IRunner
    prompts: list[str],
    cfg_scales: list[float],
    clip_scorer,                     # CLIPScorer
    attr_scorer,                     # AttributeBindingScorer
    rel_scorer,                      # RelationAccuracyScorer
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> CFGSweepResult:
    """
    Run a CFG sweep for one pipeline over a set of prompts.

    For each cfg_scale:
      1. Encode all prompts with the pipeline
      2. Generate images at that CFG scale
      3. Score with CLIPScore, attribute binding, and relation accuracy
      4. Append aggregate scores to the result

    Parameters
    ----------
    pipeline : ConditioningPipeline
    runner : T2IRunner
    prompts : list[str]
    cfg_scales : list[float]
    clip_scorer : CLIPScorer
    attr_scorer : AttributeBindingScorer
    rel_scorer : RelationAccuracyScorer
    seed : int
        Fixed seed so the only variable is CFG scale.
    output_dir : str | None
        If provided, generated images are saved to subdirectories.

    Returns
    -------
    CFGSweepResult
    """
    import os
    from pathlib import Path

    result = CFGSweepResult(pipeline_name=pipeline.name, cfg_scales=cfg_scales)

    logger.info("[%s] Starting CFG sweep over scales: %s", pipeline.name, cfg_scales)

    # Encode once — same embeddings used for all CFG scales
    logger.info("[%s] Encoding %d prompts...", pipeline.name, len(prompts))
    encoding_results = pipeline.encode_batch(prompts)
    embeddings = [r.embedding for r in encoding_results]
    seeds = [seed] * len(prompts)

    for scale in cfg_scales:
        logger.info("[%s] Generating at CFG scale %.1f", pipeline.name, scale)

        images: list[Image.Image] = runner.generate_batch(
            prompt_embeds_list=embeddings,
            cfg_scale=scale,
            seeds=seeds,
        )

        if output_dir:
            scale_dir = Path(output_dir) / pipeline.name / f"cfg{scale}"
            scale_dir.mkdir(parents=True, exist_ok=True)
            for i, (img, p) in enumerate(zip(images, prompts)):
                img.save(scale_dir / f"prompt_{i:03d}.png")

        clip_score = clip_scorer.mean_score(images, prompts)
        attr_acc = attr_scorer.mean_accuracy(images, prompts)
        rel_acc = rel_scorer.mean_accuracy(images, prompts)

        result.clip_scores.append(clip_score)
        result.attr_accuracies.append(attr_acc)
        result.rel_accuracies.append(rel_acc)

        logger.info(
            "[%s] CFG=%.1f | CLIP=%.4f | Attr=%.4f | Rel=%.4f",
            pipeline.name, scale, clip_score, attr_acc, rel_acc,
        )

    logger.info(
        "[%s] Sweep done | CLIPVar=%.6f | AttrVar=%.6f | RelVar=%.6f | Stability=%.4f",
        pipeline.name,
        result.clip_score_variance,
        result.attr_accuracy_variance,
        result.rel_accuracy_variance,
        result.compositional_stability,
    )

    return result


def compare_stability(results: list[CFGSweepResult]) -> dict[str, float]:
    """
    Compare compositional stability across pipelines.

    Returns a flat dict ready for W&B logging, plus a ranking by stability.
    """
    ranked = sorted(results, key=lambda r: r.compositional_stability, reverse=True)
    out = {}
    for rank, r in enumerate(ranked, 1):
        out[f"{r.pipeline_name}/stability_rank"] = rank
        out.update(r.to_dict())
    return out

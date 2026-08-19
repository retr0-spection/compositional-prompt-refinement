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


def _nanvar(vals: list[float]) -> float:
    """Variance over non-NaN entries; 0.0 if fewer than two remain."""
    clean = [v for v in vals if v == v]  # v == v is False only for NaN
    return float(np.var(clean)) if len(clean) >= 2 else 0.0


def _coef_var(vals: list[float]) -> Optional[float]:
    """
    Coefficient of variation (std / |mean|) over non-NaN entries.

    Returns None if fewer than two valid entries or the mean is ~0, so the
    caller can skip a metric that carries no usable spread information.
    """
    clean = [v for v in vals if v == v]
    if len(clean) < 2:
        return None
    mean = sum(clean) / len(clean)
    if abs(mean) < 1e-8:
        return None
    return float(np.std(clean) / abs(mean))


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
        return _nanvar(self.clip_scores)

    @property
    def attr_accuracy_variance(self) -> float:
        return _nanvar(self.attr_accuracies)

    @property
    def rel_accuracy_variance(self) -> float:
        return _nanvar(self.rel_accuracies)

    @property
    def compositional_stability(self) -> float:
        """
        Stability across CFG scales, higher = less sensitive.

        Defined as 1 - mean(coefficient of variation) over the available
        metrics. Using the coefficient of variation (std / |mean|) instead of
        raw variance makes the three metrics comparable despite living on very
        different scales (CLIPScore ~0.18, accuracies ~0.1) — otherwise raw
        variance lets whichever metric has the largest absolute values dominate
        the score, which is the bug that produced near-identical ~0.9999
        stability for every pipeline.

        Metrics that are entirely NaN (no applicable prompts) are skipped.
        """
        cvs = [
            _coef_var(vals)
            for vals in (self.clip_scores, self.attr_accuracies, self.rel_accuracies)
        ]
        cvs = [c for c in cvs if c is not None]
        if not cvs:
            return float("nan")
        return 1.0 - sum(cvs) / len(cvs)

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

    # Encode once — reused across all CFG scales (only needed for the legacy
    # embedding-in runner; the v2 text-in runner encodes per generate call).
    logger.info("[%s] Encoding %d prompts...", pipeline.name, len(prompts))
    encoding_results = pipeline.encode_batch(prompts)
    embeddings = [r.embedding for r in encoding_results]
    rewritten = [r.rewritten_prompt for r in encoding_results]
    seeds = [seed] * len(prompts)
    _v2_runner = hasattr(runner, "_encode")

    for scale in cfg_scales:
        logger.info("[%s] Generating at CFG scale %.1f", pipeline.name, scale)

        # Backbone-agnostic: v2 runner takes rewritten text, legacy takes embeds.
        if _v2_runner:
            images: list[Image.Image] = runner.generate_batch(
                prompts=rewritten,
                cfg_scale=scale,
                seeds=seeds,
            )
        else:
            images = runner.generate_batch(
                prompt_embeds_list=embeddings,
                cfg_scale=scale,
                seeds=seeds,
            )

        if output_dir:
            scale_dir = Path(output_dir) / pipeline.name / f"cfg{scale}"
            scale_dir.mkdir(parents=True, exist_ok=True)
            for i, (img, p) in enumerate(zip(images, prompts)):
                img.save(scale_dir / f"prompt_{i:03d}_cfg{scale:g}_seed{seed}.png")

        # CLIPScore applies to every prompt.
        clip_score = clip_scorer.mean_score(images, prompts)

        # Attribute / relation accuracy must be averaged ONLY over prompts that
        # actually contain an attribute / relation. Averaging structural zeros
        # (a spatial prompt has no colour to bind) corrupts the mean and was
        # the cause of the uniform 0.0000 relation accuracy in earlier runs.
        attr_vals, rel_vals = [], []
        for img, prompt in zip(images, prompts):
            a = attr_scorer.score(img, prompt)
            r = rel_scorer.score(img, prompt)
            if a["n_pairs"] > 0:
                attr_vals.append(a["accuracy"])
            if r["n_relations"] > 0:
                rel_vals.append(r["accuracy"])

        attr_acc = sum(attr_vals) / len(attr_vals) if attr_vals else float("nan")
        rel_acc = sum(rel_vals) / len(rel_vals) if rel_vals else float("nan")

        result.clip_scores.append(clip_score)
        result.attr_accuracies.append(attr_acc)
        result.rel_accuracies.append(rel_acc)

        logger.info(
            "[%s] CFG=%.1f | CLIP=%.4f | Attr=%s (n=%d) | Rel=%s (n=%d)",
            pipeline.name, scale, clip_score,
            f"{attr_acc:.4f}" if attr_vals else "n/a", len(attr_vals),
            f"{rel_acc:.4f}" if rel_vals else "n/a", len(rel_vals),
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

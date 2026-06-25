"""
experiments/rq4_mechanism.py

RQ4: Does the diffusion mechanism produce stronger improvements than
autoregressive rewriting under matched expansion instructions?

Compares AR pipelines (OllamaRewriter) vs LLaDA pipelines side-by-side
on identical prompt sets using the same expansion instruction.
Both use the same encoder to isolate the generative mechanism as the variable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


def run_rq4(
    ar_pipelines: list,       # list[ARPipeline] — one per encoder
    llada_pipelines: list,    # list[LLaDAPipeline] — one per encoder
    runner,
    prompt_sets: dict[str, list[str]],
    clip_scorer,
    attr_scorer,
    rel_scorer,
    seed: int = 42,
    cfg_scale: float = 7.5,
    output_dir: str | Path = "outputs/rq4",
    wandb_log: bool = True,
) -> dict:
    """
    Run RQ4 mechanism comparison.

    AR pipelines (Ollama) and LLaDA pipelines are matched by encoder:
        ar_clip   vs llada_clip
        ar_longclip  vs llada_longclip

    The expansion instruction is identical for both mechanisms (controlled
    in the rewriter implementations), so any metric difference is attributable
    to the generative mechanism (left-to-right AR vs masked diffusion).

    Returns
    -------
    dict with 'ar', 'llada', and 'comparison' sub-dicts.
    """
    from evaluation.metrics import score_all
    from evaluation.embedding_analysis import analyse_semantic_density
    from utils.logging import log_metrics

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pipelines = ar_pipelines + llada_pipelines
    all_prompts = []
    for prompts in prompt_sets.values():
        all_prompts.extend(prompts)
    # Deduplicate while preserving order
    seen = set()
    flat_prompts = [p for p in all_prompts if not (p in seen or seen.add(p))]

    results: dict = {"ar": {}, "llada": {}, "comparison": {}}

    for pipeline in all_pipelines:
        mechanism = "ar" if "ar_" in pipeline.name else "llada"
        logger.info("[RQ4] Pipeline: %s (mechanism: %s)", pipeline.name, mechanism)

        for set_name, prompts in prompt_sets.items():
            logger.info("[RQ4][%s] Processing '%s' (%d prompts)",
                        pipeline.name, set_name, len(prompts))

            encoding_results = pipeline.encode_batch(prompts)
            embeddings = [r.embedding for r in encoding_results]
            raw_prompts = [r.raw_prompt for r in encoding_results]
            rewritten_prompts = [r.rewritten_prompt for r in encoding_results]

            # Expansion quality: semantic density
            density_stats = analyse_semantic_density(
                raw_prompts=raw_prompts,
                rewritten_prompts=rewritten_prompts,
                pipeline_name=pipeline.name,
            )

            # Generate images
            images = runner.generate_batch(
                prompt_embeds_list=embeddings,
                cfg_scale=cfg_scale,
                seeds=[seed] * len(prompts),
            )

            # Save images
            img_dir = output_dir / pipeline.name / set_name
            img_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(images):
                img.save(img_dir / f"prompt_{i:03d}.png")

            # Score
            eval_result = score_all(
                pipeline_name=pipeline.name,
                images=images,
                prompts=prompts,
                clip_scorer=clip_scorer,
                attr_scorer=attr_scorer,
                rel_scorer=rel_scorer,
            )

            key = f"{pipeline.name}/{set_name}"
            metrics = {
                f"{key}/clip_score": eval_result.clip_score,
                f"{key}/attr_binding_accuracy": eval_result.attr_binding_accuracy,
                f"{key}/relation_accuracy": eval_result.relation_accuracy,
                **density_stats,
            }
            results[mechanism].setdefault(pipeline.name, {})[set_name] = metrics

            if wandb_log:
                log_metrics(metrics)

            logger.info(
                "[RQ4][%s][%s] CLIP=%.4f | Attr=%.4f | Rel=%.4f",
                pipeline.name, set_name,
                eval_result.clip_score,
                eval_result.attr_binding_accuracy,
                eval_result.relation_accuracy,
            )

    # Head-to-head comparison: AR vs LLaDA per encoder
    results["comparison"] = _compare_mechanisms(results["ar"], results["llada"])

    if wandb_log:
        log_metrics(results["comparison"])

    _save_summary(results, output_dir / "rq4_summary.txt")
    return results


def _compare_mechanisms(
    ar_results: dict,
    llada_results: dict,
) -> dict[str, float]:
    """
    Compute per-metric deltas: LLaDA score − AR score.

    Positive delta means LLaDA outperforms AR.
    """
    comparison = {}
    for ar_name, ar_sets in ar_results.items():
        # Match to corresponding LLaDA pipeline by encoder name
        encoder = ar_name.replace("ar_", "")
        llada_name = f"llada_{encoder}"
        llada_sets = llada_results.get(llada_name, {})

        for set_name in ar_sets:
            ar_metrics = ar_sets.get(set_name, {})
            llada_metrics = llada_sets.get(set_name, {})

            for metric in ["clip_score", "attr_binding_accuracy", "relation_accuracy"]:
                ar_key = f"{ar_name}/{set_name}/{metric}"
                llada_key = f"{llada_name}/{set_name}/{metric}"
                ar_val = ar_metrics.get(ar_key, 0.0)
                llada_val = llada_metrics.get(llada_key, 0.0)
                delta_key = f"delta_{encoder}/{set_name}/{metric}"
                comparison[delta_key] = llada_val - ar_val

    return comparison


def _save_summary(results: dict, path: Path) -> None:
    with open(path, "w") as f:
        f.write("RQ4 — Mechanism Comparison (AR vs LLaDA)\n")
        f.write("=" * 50 + "\n\n")
        for mechanism in ["ar", "llada"]:
            f.write(f"Mechanism: {mechanism.upper()}\n")
            for pipeline_name, set_results in results[mechanism].items():
                f.write(f"  Pipeline: {pipeline_name}\n")
                for set_name, metrics in set_results.items():
                    f.write(f"    Set: {set_name}\n")
                    for k, v in sorted(metrics.items()):
                        if isinstance(v, float):
                            f.write(f"      {k}: {v:.4f}\n")
            f.write("\n")
        f.write("Mechanism Deltas (LLaDA − AR, positive = LLaDA wins):\n")
        for k, v in sorted(results["comparison"].items()):
            f.write(f"  {k}: {v:+.4f}\n")
    logger.info("RQ4 summary saved to %s", path)

"""
experiments/rq2_compositional.py

RQ2: Does improved conditioning reduce compositional errors?

For each pipeline × encoder × seed:
  1. Encode all prompts in the eval sets
  2. Generate images (SD 2.1, default CFG scale)
  3. Score with CLIPScore, attribute binding accuracy, relation accuracy
  4. Compute FID against a reference set (if provided)

Results are logged to W&B and saved to output_dir.
Each pipeline×set also writes a trace.jsonl with full per-prompt traceability:
    raw_prompt → rewritten_prompt → token counts → image path → per-prompt scores
"""

from __future__ import annotations

import json
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
    unload_runner_before_scoring: bool = True,
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
    seed : int
    cfg_scale : float
    output_dir : str | Path
    wandb_log : bool
    reference_images : list[PIL.Image] | None
        Explicit FID reference. If None, the first pipeline's images are used.
    unload_runner_before_scoring : bool
        Unload SD 2.1 before loading VQA scorers to avoid OOM on small GPUs.

    Returns
    -------
    dict mapping pipeline_name → {set_name: metric_dict}
    """
    from evaluation.metrics import EvalResult
    from utils.logging import log_metrics
    from utils.naming import img_name, write_meta, check_meta

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    _fid_reference: Optional[list[Image.Image]] = reference_images
    _raw_pipeline_images: list[Image.Image] = []

    if fid_scorer and _fid_reference:
        logger.info("Loading %d explicit reference images into FID scorer", len(_fid_reference))
        fid_scorer.update_real(_fid_reference)

    # -----------------------------------------------------------------------
    # Phase 1: Generate all images (runner in memory, scorers not yet loaded)
    # Caches both images and encoding results for traceability.
    # -----------------------------------------------------------------------
    # generated_cache[pipeline_name][set_name] = list[PIL.Image]
    # encoding_cache[pipeline_name][set_name]  = list[EncodingResult]
    generated_cache: dict[str, dict[str, list[Image.Image]]] = {}
    encoding_cache: dict[str, dict[str, list]] = {}

    for pipeline in pipelines:
        logger.info("[RQ2] Phase 1 — generating | Pipeline: %s | CFG=%.1f | Seed=%d",
                    pipeline.name, cfg_scale, seed)
        generated_cache[pipeline.name] = {}
        encoding_cache[pipeline.name] = {}

        for set_name, prompts in prompt_sets.items():
            logger.info("[RQ2][%s] Encoding %d prompts from '%s'",
                        pipeline.name, len(prompts), set_name)
            enc_results = pipeline.encode_batch(prompts)

            img_dir = output_dir / pipeline.name / set_name
            img_dir.mkdir(parents=True, exist_ok=True)

            # Run-level manifest: warn loudly if cached images were produced
            # under different settings (model, steps, resolution, ...).
            run_settings = {
                "cfg_scale": cfg_scale,
                "seed": seed,
                "model_id": getattr(runner, "model_id", "unknown"),
                "num_inference_steps": getattr(runner, "num_inference_steps", "unknown"),
                "resolution": getattr(runner, "resolution", "unknown"),
            }
            check_meta(img_dir, run_settings)
            write_meta(img_dir, run_settings)

            # ---- Disk resume: load existing images, generate only missing ----
            img_paths = [img_dir / img_name(i, cfg_scale, seed) for i in range(len(prompts))]
            images: list[Optional[Image.Image]] = [None] * len(prompts)
            missing: list[int] = []
            for i, p in enumerate(img_paths):
                if p.exists():
                    try:
                        images[i] = Image.open(p).convert("RGB")
                    except Exception:
                        logger.warning("[RQ2][%s] Corrupt image %s — regenerating.",
                                       pipeline.name, p)
                        missing.append(i)
                else:
                    missing.append(i)

            if missing:
                logger.info(
                    "[RQ2][%s] '%s': %d/%d images cached on disk, generating %d...",
                    pipeline.name, set_name,
                    len(prompts) - len(missing), len(prompts), len(missing),
                )
                new_images = runner.generate_batch(
                    prompt_embeds_list=[enc_results[i].embedding for i in missing],
                    cfg_scale=cfg_scale,
                    seeds=[seed] * len(missing),
                )
                for i, img in zip(missing, new_images):
                    img.save(img_paths[i])
                    images[i] = img
            else:
                logger.info("[RQ2][%s] '%s': all %d images cached on disk — skipping generation.",
                            pipeline.name, set_name, len(prompts))

            generated_cache[pipeline.name][set_name] = images
            encoding_cache[pipeline.name][set_name] = enc_results

            if fid_scorer and _fid_reference is None:
                _raw_pipeline_images.extend(images)

        if fid_scorer and _fid_reference is None and _raw_pipeline_images:
            _fid_reference = list(_raw_pipeline_images)
            _raw_pipeline_images.clear()
            logger.info(
                "Using %d images from '%s' as FID reference for remaining pipelines.",
                len(_fid_reference), pipeline.name,
            )
            fid_scorer.update_real(_fid_reference)

    # -----------------------------------------------------------------------
    # Phase 2: Unload runner, then score with full per-prompt tracing.
    # -----------------------------------------------------------------------
    if unload_runner_before_scoring:
        logger.info("[RQ2] Unloading T2IRunner before scoring to free memory.")
        runner.unload()

    for pipeline in pipelines:
        pipeline_results: dict[str, dict] = {}

        for set_name, prompts in prompt_sets.items():
            images = generated_cache[pipeline.name][set_name]
            enc_results = encoding_cache[pipeline.name][set_name]
            img_dir = output_dir / pipeline.name / set_name
            trace_path = img_dir / "trace.jsonl"

            # ---- Scoring resume: skip set if a complete trace already exists ----
            # (FID is excluded from this skip — it is cross-set and cheap relative
            # to VQA scoring; it recomputes below only when fid_scorer is given.)
            existing_trace = _load_trace(trace_path)
            if existing_trace is not None and len(existing_trace) == len(prompts):
                logger.info(
                    "[RQ2][%s] '%s': complete trace.jsonl found (%d records) — "
                    "reusing scores, skipping VQA/CLIP scoring.",
                    pipeline.name, set_name, len(existing_trace),
                )
                clip_scores = [r["clip_score"] for r in existing_trace]
                attr_accs = [r["attr_binding"]["accuracy"] for r in existing_trace]
                rel_accs = [r["relation_accuracy"]["accuracy"] for r in existing_trace]
                trace_records = existing_trace
            else:
                # Per-prompt scoring — collects detail for trace and aggregates
                clip_scores, attr_accs, rel_accs = [], [], []
                trace_records = []

                for i, (img, prompt, enc) in enumerate(zip(images, prompts, enc_results)):
                    clip_s = clip_scorer.score(img, prompt)
                    attr_r = attr_scorer.score(img, prompt)
                    rel_r  = rel_scorer.score(img, prompt)

                    clip_scores.append(clip_s)
                    attr_accs.append(attr_r["accuracy"])
                    rel_accs.append(rel_r["accuracy"])

                    trace_records.append({
                        "idx": i,
                        "pipeline": pipeline.name,
                        "set": set_name,
                        "raw_prompt": enc.raw_prompt,
                        "rewritten_prompt": enc.rewritten_prompt,
                        "token_count_raw": enc.token_count_raw,
                        "token_count_rewritten": enc.token_count_rewritten,
                        "was_truncated": enc.was_truncated,
                        "image_path": str(img_dir / img_name(i, cfg_scale, seed)),
                        "clip_score": clip_s,
                        "attr_binding": attr_r,
                        "relation_accuracy": rel_r,
                    })

                # Write per-prompt trace
                _write_trace(trace_records, trace_path)

            n = len(prompts) or 1
            result = EvalResult(
                pipeline_name=pipeline.name,
                clip_score=sum(clip_scores) / n,
                attr_binding_accuracy=sum(attr_accs) / n,
                relation_accuracy=sum(rel_accs) / n,
            )

            # FID
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

    _save_summary(all_results, output_dir / "rq2_summary.txt")
    return all_results


def _write_trace(records: list[dict], path: Path) -> None:
    """Write per-prompt trace records to a JSONL file (one JSON object per line)."""
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")
    logger.debug("Trace written to %s (%d records)", path, len(records))


def _load_trace(path: Path) -> Optional[list[dict]]:
    """Load an existing trace.jsonl; return None if absent or unreadable."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        logger.warning("Could not load trace %s (%s) — will re-score.", path, exc)
        return None


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

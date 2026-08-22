"""
experiments/cfg_stability.py

CFG-stability-across-denoising-steps diagnostic.

Distinct from RQ3 (which measures variance of the FINAL image quality across
CFG SCALES), this measures how STABLE the denoising TRAJECTORY is at each CFG
scale — i.e. does strong guidance make the per-step path erratic (the metric
jumps around from step to step) versus low guidance producing a smooth climb?

Motivation
----------
CFG amplifies the conditioning direction at every denoising step. Too much
guidance is known to over-saturate and destabilise sampling. This experiment
asks, per conditioning pipeline: at which CFG scale does the denoising
trajectory stay smooth, and where does it start to thrash? A pipeline whose
trajectory stays stable across a wider CFG range is more robust to guidance.

    stability(scale) = 1 / (1 + mean step-to-step |Δmetric|)

Higher = smoother trajectory at that scale. Aggregated over the prompt subset.

Scope & cost
------------
n_prompts × n_cfg_scales × (steps / score_every) decode+score ops. Default
8 prompts × 5 scales × ~10 scored steps = ~400 ops per pipeline: a few minutes
on the A6000. This is a diagnostic on a SMALL subset, not a benchmark. Metric
is CLIPScore only (BLIP-2 is too slow per-step).

Output
------
outputs/cfg_stability/<pipeline>/trajectory_cfg.jsonl
    one record per (prompt, cfg_scale, step): {idx, cfg, step, clip_score}
outputs/cfg_stability/<pipeline>/stability.json
    {cfg_scale: stability_value} aggregated over prompts

Usage
-----
    python experiments/cfg_stability.py
    python experiments/cfg_stability.py --n-prompts 6 --pipeline llada_clip
    python experiments/cfg_stability.py --set color_binding --no-wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _trajectory_stability(scores_by_step: list[float]) -> float:
    """
    Smoothness of a single denoising trajectory.

    stability = 1 / (1 + mean |Δ| between consecutive scored steps).
    A monotonic smooth climb -> small deltas -> stability near 1.
    A jumpy/oscillating path -> large deltas -> stability toward 0.
    """
    if len(scores_by_step) < 2:
        return float("nan")
    deltas = [abs(scores_by_step[i] - scores_by_step[i - 1])
              for i in range(1, len(scores_by_step))]
    mean_abs_delta = sum(deltas) / len(deltas)
    return 1.0 / (1.0 + mean_abs_delta)


def run_cfg_stability(
    pipelines: list,
    runner,
    clip_scorer,
    prompts: list[str],
    cfg_scales: list[float],
    seed: int = 42,
    score_every: int = 5,
    output_dir: str | Path = "outputs/cfg_stability",
    wandb_log: bool = True,
) -> dict:
    """
    For each pipeline × cfg_scale, generate the prompt subset with per-step
    scoring and measure trajectory stability. Returns
    {pipeline: {cfg_scale: mean_stability}}.
    """
    output_dir = Path(output_dir)
    v2_runner = hasattr(runner, "_encode")   # text-in runner vs legacy embedding-in
    all_stability: dict[str, dict[float, float]] = {}

    for pipeline in pipelines:
        cell_dir = output_dir / pipeline.name
        cell_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict] = []
        # Rewrite once per prompt (cached); reused across all CFG scales.
        enc_results = pipeline.encode_batch(prompts)

        per_scale_stability: dict[float, list[float]] = {s: [] for s in cfg_scales}

        for scale in cfg_scales:
            logger.info("[cfg-stab][%s] CFG=%.1f over %d prompts",
                        pipeline.name, scale, len(prompts))
            for idx, enc in enumerate(enc_results):
                # Backbone-agnostic trajectory call.
                if v2_runner:
                    _, traj = runner.generate_with_trajectory(
                        prompt_text=enc.rewritten_prompt,
                        clip_scorer=clip_scorer,
                        cfg_scale=scale, seed=seed, score_every=score_every,
                    )
                else:
                    _, traj = runner.generate_with_trajectory(
                        prompt_embeds=enc.embedding,
                        prompt_text=enc.raw_prompt,
                        clip_scorer=clip_scorer,
                        cfg_scale=scale, seed=seed, score_every=score_every,
                    )

                scores = [pt["clip_score"] for pt in traj]
                stab = _trajectory_stability(scores)
                if stab == stab:  # not NaN
                    per_scale_stability[scale].append(stab)

                for pt in traj:
                    records.append({
                        "idx": idx, "cfg": scale,
                        "step": pt["step"], "clip_score": pt["clip_score"],
                    })

        # Aggregate stability per scale (mean over prompts).
        stability = {
            scale: (sum(v) / len(v) if v else float("nan"))
            for scale, v in per_scale_stability.items()
        }
        all_stability[pipeline.name] = stability

        # Persist per-step records and the stability summary.
        with open(cell_dir / "trajectory_cfg.jsonl", "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(cell_dir / "stability.json", "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in stability.items()}, f, indent=2)

        logger.info("[cfg-stab][%s] stability by CFG: %s", pipeline.name,
                    {k: round(v, 4) for k, v in stability.items()})

        if wandb_log:
            from utils.logging import log_metrics
            log_metrics({f"cfg_stability/{pipeline.name}/cfg{scale:g}": v
                         for scale, v in stability.items()})

    return all_stability


def main() -> None:
    parser = argparse.ArgumentParser(description="CFG stability across denoising steps")
    parser.add_argument("--n-prompts", type=int, default=8,
                        help="Prompts per pipeline (keep small — per-step scoring is slow)")
    parser.add_argument("--score-every", type=int, default=5)
    parser.add_argument("--set", default="color_binding", metavar="NAME",
                        help="Prompt set to sample from (default: color_binding)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pipeline", default=None, metavar="NAME")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--config", nargs="*", metavar="KEY=VALUE")
    args = parser.parse_args()

    from experiments.run_experiment import (
        load_config, build_pipelines, build_runner, build_scorers,
    )
    from utils.prompt_io import load_prompts
    from utils.seed import set_seed

    cfg = load_config(args.config)
    seed = args.seed or cfg["seeds"][0]
    cfg_scales = cfg["cfg_scales"]
    wandb_log = not args.no_wandb
    set_seed(seed)

    if wandb_log:
        from utils.logging import init_wandb
        suffix = f"_{args.pipeline}" if args.pipeline else ""
        init_wandb(
            project=cfg.get("wandb_project", "prompt-pipeline"),
            run_name=f"cfg_stability_seed{seed}{suffix}",
            config={**cfg, "seed": seed, "score_every": args.score_every,
                    "n_prompts": args.n_prompts},
            offline=cfg.get("wandb_offline", False),
        )

    pipelines = build_pipelines(cfg, dry_run=False)
    if args.pipeline:
        pipelines = [p for p in pipelines if p.name == args.pipeline]
        if not pipelines:
            raise ValueError(f"--pipeline {args.pipeline!r} not found.")

    runner = build_runner(cfg, dry_run=False)
    clip_scorer, _, _, _ = build_scorers(cfg, dry_run=False)
    prompts = load_prompts(args.set)[:args.n_prompts]

    logger.info("CFG-stability | pipelines=%s | %d prompts | scales=%s",
                [p.name for p in pipelines], len(prompts), cfg_scales)

    run_cfg_stability(
        pipelines=pipelines, runner=runner, clip_scorer=clip_scorer,
        prompts=prompts, cfg_scales=cfg_scales, seed=seed,
        score_every=args.score_every,
        output_dir=Path(cfg.get("output_dir", "outputs")) / "cfg_stability",
        wandb_log=wandb_log,
    )

    if wandb_log:
        from utils.logging import finish
        finish()
    logger.info("CFG-stability experiment done.")


if __name__ == "__main__":
    main()

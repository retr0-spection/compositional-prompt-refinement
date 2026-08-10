"""
experiments/trajectory.py

Denoising-trajectory experiment: measures how prompt-image alignment
(CLIPScore) builds up ACROSS diffusion timesteps, rather than only at the
final image.

Motivation
----------
The proposal's H1 claims refined prompts guide denoising along "more
compositionally aligned trajectories". Every other RQ measures the destination
(the final image); this measures the path. For each pipeline it produces a
CLIPScore-vs-denoising-step curve, so you can ask whether refined conditioning
reaches high alignment earlier / higher / more smoothly than raw conditioning.

Scope & cost
------------
Each scored step adds a VAE decode + a CLIP forward pass, so this runs ~10x
slower per image than normal generation. It therefore uses a SMALL diagnostic
subset (default 8 prompts per pipeline) — enough to see the trajectory shape,
not a full benchmark. The metric is CLIPScore only; BLIP-2 attribute/relation
scoring is far too slow to run per-step, so trajectories speak to semantic
alignment (RQ1's "aligned trajectories"), NOT to attribute binding directly.
State that limitation when reporting.

Output
------
outputs/trajectory/<pipeline>/<set>/trajectory.jsonl
    one record per (prompt, scored step): {idx, step, clip_score, prompt}
which evaluation.plotting.plot_trajectory() reads directly.

Usage
-----
    python experiments/trajectory.py
    python experiments/trajectory.py --n-prompts 6 --score-every 5
    python experiments/trajectory.py --set color_binding --no-wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_trajectory(
    pipelines: list,
    runner,
    clip_scorer,
    prompt_sets: dict[str, list[str]],
    seed: int = 42,
    cfg_scale: float = 7.5,
    score_every: int = 5,
    output_dir: str | Path = "outputs/trajectory",
    wandb_log: bool = True,
) -> dict:
    """
    Run the denoising-trajectory measurement for each pipeline × set.

    Returns {pipeline: {set: [records]}} and writes trajectory.jsonl per cell.
    """
    output_dir = Path(output_dir)
    all_traj: dict[str, dict[str, list[dict]]] = {}

    for pipeline in pipelines:
        all_traj[pipeline.name] = {}
        for set_name, prompts in prompt_sets.items():
            logger.info("[traj][%s] set '%s' — %d prompts, scoring every %d steps",
                        pipeline.name, set_name, len(prompts), score_every)

            enc_results = pipeline.encode_batch(prompts)
            records: list[dict] = []

            for idx, (prompt, enc) in enumerate(zip(prompts, enc_results)):
                _, trajectory = runner.generate_with_trajectory(
                    prompt_embeds=enc.embedding,
                    prompt_text=enc.raw_prompt,   # score vs. user intent
                    clip_scorer=clip_scorer,
                    cfg_scale=cfg_scale,
                    seed=seed,
                    score_every=score_every,
                )
                for point in trajectory:
                    records.append({
                        "idx": idx,
                        "step": point["step"],
                        "clip_score": point["clip_score"],
                        "prompt": enc.raw_prompt,
                    })
                logger.info("[traj][%s][%s] prompt %d/%d done (%d points)",
                            pipeline.name, set_name, idx + 1, len(prompts),
                            len(trajectory))

            # Persist where plotting.load_trajectory expects it.
            cell_dir = output_dir / pipeline.name / set_name
            cell_dir.mkdir(parents=True, exist_ok=True)
            traj_path = cell_dir / "trajectory.jsonl"
            with open(traj_path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info("[traj][%s][%s] wrote %d records → %s",
                        pipeline.name, set_name, len(records), traj_path)

            all_traj[pipeline.name][set_name] = records

    # Render the plots immediately (also logs to W&B if a run is active).
    from evaluation.plotting import plot_trajectory
    plot_dir = Path(output_dir).parent / "plots"
    plot_trajectory(output_dir, plot_dir, metric="clip_score", set_name=None)
    for set_name in prompt_sets:
        plot_trajectory(output_dir, plot_dir, metric="clip_score", set_name=set_name)

    return all_traj


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoising-trajectory experiment")
    parser.add_argument("--n-prompts", type=int, default=8,
                        help="Prompts per pipeline×set (keep small — this is ~10x slow)")
    parser.add_argument("--score-every", type=int, default=5,
                        help="Score the latent once per this many denoising steps")
    parser.add_argument("--set", default=None, metavar="NAME",
                        help="Restrict to a single prompt set (default: all rq2 sets)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--pipeline", default=None, metavar="NAME",
                        help="Restrict to one pipeline (e.g. llada_clip)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--config", nargs="*", metavar="KEY=VALUE")
    args = parser.parse_args()

    # Reuse the main experiment's builders so config handling is identical.
    from experiments.run_experiment import (
        load_config, build_pipelines, build_runner, build_scorers,
    )
    from utils.prompt_io import load_prompts
    from utils.seed import set_seed

    cfg = load_config(args.config)
    seed = args.seed or cfg["seeds"][0]
    cfg_scale = args.cfg or cfg["default_cfg_scale"]
    wandb_log = not args.no_wandb
    set_seed(seed)

    if wandb_log:
        from utils.logging import init_wandb
        suffix = f"_{args.pipeline}" if args.pipeline else ""
        init_wandb(
            project=cfg.get("wandb_project", "prompt-pipeline"),
            run_name=f"trajectory_seed{seed}{suffix}",
            config={**cfg, "seed": seed, "cfg_scale": cfg_scale,
                    "score_every": args.score_every, "n_prompts": args.n_prompts},
            offline=cfg.get("wandb_offline", False),
        )

    pipelines = build_pipelines(cfg, dry_run=False)
    if args.pipeline:
        pipelines = [p for p in pipelines if p.name == args.pipeline]
        if not pipelines:
            raise ValueError(f"--pipeline {args.pipeline!r} not found.")

    runner = build_runner(cfg, dry_run=False)
    clip_scorer, _, _, _ = build_scorers(cfg, dry_run=False)

    # Which sets: default to RQ2's sets, truncated to n_prompts each.
    set_names = [args.set] if args.set else cfg["eval_prompt_sets"]["rq2"]
    prompt_sets = {s: load_prompts(s)[:args.n_prompts] for s in set_names}

    logger.info("Trajectory experiment | pipelines=%s | sets=%s | %d prompts each",
                [p.name for p in pipelines], set_names, args.n_prompts)

    run_trajectory(
        pipelines=pipelines, runner=runner, clip_scorer=clip_scorer,
        prompt_sets=prompt_sets, seed=seed, cfg_scale=cfg_scale,
        score_every=args.score_every,
        output_dir=Path(cfg.get("output_dir", "outputs")) / "trajectory",
        wandb_log=wandb_log,
    )

    if wandb_log:
        from utils.logging import finish
        finish()
    logger.info("Trajectory experiment done.")


if __name__ == "__main__":
    main()

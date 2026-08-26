"""
experiments/tunability.py

RQ6: Tunability sweep across BOTH layers of the pipeline.

A 16x16 = 256-cell crossed grid over two hyperparameters per layer:
    LLaDA (language) axis : steps x gen_length      (controls rewrite cost/richness)
    image (diffusion) axis: num_inference_steps x cfg_scale (controls gen cost/steer)

For each cell we measure:
    - time      : LLaDA rewrite seconds + image generation seconds
    - quality   : CLIPScore (cheap, per-image; BLIP-2 too slow for 256 cells)
    - steerability (post-hoc): how much quality changes ALONG each axis, i.e.
      the gradient of the metric surface — computed after the grid is filled.

Scope: a SMALL prompt subset (~15) held fixed across all 256 cells. This is a
tuning study mapping the cost/quality/steerability surface, not a benchmark;
per-cell values are coarse (few prompts), the SHAPE of the surface is the result.

LLaDA rewrites are cached per (prompt, llada-config): 16 LLaDA configs x 15
prompts = 240 unique rewrites computed once, reused across all 16 image configs.
Image gen is the bulk: 256 x 15 = 3840 generations.

Output: outputs/<backbone>/rq6/grid.jsonl (one row per cell) + heatmaps.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _build_axes(n: int = 16) -> tuple[list[dict], list[dict]]:
    """
    Build the two 16-point axes as 4x4 grids over two knobs each.

    LLaDA axis:  steps in {32,64,96,128} x gen_length in {32,64,96,128}
    image axis:  num_inference_steps in {10,25,50,100} x cfg in {2,5,7.5,12}
    """
    llada_steps = [32, 64, 96, 128]
    llada_genlen = [32, 64, 96, 128]
    img_steps = [10, 25, 50, 100]
    img_cfg = [2.0, 5.0, 7.5, 12.0]

    llada_axis = [{"steps": s, "gen_length": g}
                  for s in llada_steps for g in llada_genlen]
    image_axis = [{"num_inference_steps": s, "cfg_scale": c}
                  for s in img_steps for c in img_cfg]
    return llada_axis[:n], image_axis[:n]


def run_tunability(
    prompts: list[str],
    llada_rewriter,
    runner,
    clip_scorer,
    seed: int = 42,
    output_dir: str | Path = "outputs/rq6",
    wandb_log: bool = True,
    n_axis: int = 16,
    showcase_idx: int = 0,
) -> dict:
    """
    Run the 256-cell tunability grid over a small fixed prompt subset.

    llada_rewriter : an LLaDARewriter whose config we mutate per LLaDA cell.
    runner         : T2IRunnerV2 (text-in); num_inference_steps/cfg passed per cell.
    showcase_idx   : which prompt's image to SAVE per cell for the image-grid
                     figure (default: first prompt). One image per cell is kept
                     at rq6/grid_images/L{li}_I{ii}.png so a readable 4x4 slice
                     can be laid out later without regenerating.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_img_dir = output_dir / "grid_images"
    grid_img_dir.mkdir(parents=True, exist_ok=True)
    llada_axis, image_axis = _build_axes(n_axis)

    grid_path = output_dir / "grid.jsonl"
    # Resume: load already-computed cells (keyed by llada_idx,image_idx).
    done: set[tuple[int, int]] = set()
    if grid_path.exists():
        with open(grid_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["llada_idx"], r["image_idx"]))
        logger.info("[RQ6] Resuming — %d cells already computed.", len(done))

    grid_file = open(grid_path, "a", encoding="utf-8")

    for li, lcfg in enumerate(llada_axis):
        # Apply LLaDA config for this row. Our disk cache is keyed by prompt
        # text alone, so clear the in-memory cache per LLaDA config so different
        # configs don't collide (different configs must produce fresh rewrites).
        llada_rewriter.config.steps = lcfg["steps"]
        llada_rewriter.config.gen_length = lcfg["gen_length"]
        # LLaDA's _generate requires BOTH:
        #   gen_length % block_length == 0   AND   steps % (gen_length//block_length) == 0
        # Setting block_length = gen_length makes num_blocks = 1, which divides
        # any steps value — sidestepping the semi-autoregressive blocking, which
        # a tunability sweep over (steps, gen_length) doesn't need to vary.
        llada_rewriter.config.block_length = lcfg["gen_length"]
        llada_rewriter._cache = {}  # force re-rewrite under this config

        # Rewrite the subset once for this LLaDA config (timed).
        rewrites, rw_secs = [], []
        for p in prompts:
            t0 = time.perf_counter()
            rw = llada_rewriter.rewrite(p)
            rw_secs.append(time.perf_counter() - t0)
            rewrites.append(rw)
        mean_rw_time = sum(rw_secs) / len(rw_secs) if rw_secs else 0.0

        for ii, icfg in enumerate(image_axis):
            if (li, ii) in done:
                continue

            # Generate the subset at this image config (timed).
            t0 = time.perf_counter()
            images = runner.generate_batch(
                prompts=rewrites,
                cfg_scale=icfg["cfg_scale"],
                seeds=[seed] * len(rewrites),
                num_inference_steps=icfg["num_inference_steps"],
            )
            gen_secs = time.perf_counter() - t0
            mean_gen_time = gen_secs / len(images) if images else 0.0

            # Quality: mean CLIPScore vs the original prompts.
            clip_scores = [clip_scorer.score(img, p)
                           for img, p in zip(images, prompts)]
            mean_clip = sum(clip_scores) / len(clip_scores) if clip_scores else 0.0

            # Save the showcase prompt's image for the image-grid figure.
            if 0 <= showcase_idx < len(images):
                try:
                    images[showcase_idx].save(grid_img_dir / f"L{li}_I{ii}.png")
                except Exception as exc:
                    logger.debug("Could not save showcase image L%d_I%d (%s)", li, ii, exc)

            row = {
                "llada_idx": li, "image_idx": ii,
                "llada_steps": lcfg["steps"], "llada_gen_length": lcfg["gen_length"],
                "img_steps": icfg["num_inference_steps"], "cfg_scale": icfg["cfg_scale"],
                "mean_rewrite_time": mean_rw_time,
                "mean_gen_time": mean_gen_time,
                "total_time": mean_rw_time + mean_gen_time,
                "mean_clip_score": mean_clip,
            }
            grid_file.write(json.dumps(row) + "\n")
            grid_file.flush()
            logger.info("[RQ6] cell (L%d,I%d): clip=%.4f gen=%.2fs rw=%.2fs",
                        li, ii, mean_clip, mean_gen_time, mean_rw_time)

    grid_file.close()

    # ---- Post-hoc steerability: gradient of CLIPScore along each axis ----
    _compute_steerability(grid_path, output_dir)

    # ---- Heatmaps ----
    try:
        from evaluation.plotting import plot_tunability_heatmaps
        plot_tunability_heatmaps(output_dir, output_dir.parent / "plots")
    except Exception as exc:
        logger.warning("[RQ6] heatmap plotting failed (%s)", exc)

    logger.info("[RQ6] Tunability sweep complete: %s", grid_path)
    return {"grid_path": str(grid_path)}


def _compute_steerability(grid_path: Path, output_dir: Path) -> None:
    """
    Steerability = mean absolute gradient of CLIPScore along each axis.

    High steerability on an axis means turning that knob changes quality a lot
    (strong control); low means the knob barely matters. Computed as the mean
    |Δ CLIPScore| between adjacent cells along each axis.
    """
    import numpy as np

    rows = []
    with open(grid_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return

    n_l = max(r["llada_idx"] for r in rows) + 1
    n_i = max(r["image_idx"] for r in rows) + 1
    grid = np.full((n_l, n_i), np.nan)
    for r in rows:
        grid[r["llada_idx"], r["image_idx"]] = r["mean_clip_score"]

    # Gradient along image axis (rows fixed), and along LLaDA axis (cols fixed).
    img_grad = float(np.nanmean(np.abs(np.diff(grid, axis=1))))
    llada_grad = float(np.nanmean(np.abs(np.diff(grid, axis=0))))

    steerability = {
        "image_axis_steerability": img_grad,
        "llada_axis_steerability": llada_grad,
        "note": "mean |delta CLIPScore| between adjacent cells along each axis",
    }
    with open(output_dir / "steerability.json", "w", encoding="utf-8") as f:
        json.dump(steerability, f, indent=2)
    logger.info("[RQ6] steerability: image-axis=%.4f  llada-axis=%.4f",
                img_grad, llada_grad)

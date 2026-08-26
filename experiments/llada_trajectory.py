"""
experiments/llada_trajectory.py

Language-layer trajectory diagnostic: how a LLaDA rewrite EMERGES across the
masked-diffusion unmasking steps.

Distinct from the image-layer denoising trajectory (cfg_stability.py). This
measures the LANGUAGE side: at each unmasking step, what fraction of the
response is unmasked, and how much compositional content (attribute + relation
counts, semantic density) the partial decode already contains.

Why it matters for the thesis: it visualises the "joint resolution" that
defines the diffusion mechanism — the whole sequence is progressively resolved
rather than committed left-to-right. An autoregressive model has no analogous
trajectory (it emits final tokens one at a time), so this figure is unique to
the diffusion-LM side of the comparison and directly motivates the PoE capstone.

Scope: a SMALL prompt subset (~10-15). With --track-content it decodes and
scene-graph-counts the partial response every step (needs Ollama up; slower);
without it, only the unmasked-fraction schedule is recorded (fast, no Ollama).

Output:
    outputs/<backbone>/llada_trajectory/trajectory.jsonl   (idx, step, ...)
Then plotted by evaluation.plotting.plot_llada_trajectory.

Usage:
    python experiments/llada_trajectory.py --n-prompts 10
    python experiments/llada_trajectory.py --n-prompts 8 --track-content   # needs Ollama
    python experiments/llada_trajectory.py --set rq5_compositional --n-prompts 12
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


def run_llada_trajectory(
    prompts: list[str],
    llada_rewriter,
    extractor=None,
    output_dir: str | Path = "outputs/llada_trajectory",
    wandb_log: bool = True,
) -> dict:
    """
    Record the per-unmasking-step trajectory for each prompt.

    extractor : optional SemanticExtractor. If given, compositional content of
                the partial decode is tracked at each step (slower, needs Ollama).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for idx, prompt in enumerate(prompts):
        logger.info("[llada-traj] %d/%d: %r", idx + 1, len(prompts), prompt[:50])
        _, trajectory = llada_rewriter.rewrite_with_trace(prompt, extractor=extractor)
        for pt in trajectory:
            pt = {"idx": idx, "prompt": prompt, **pt}
            records.append(pt)

    with open(output_dir / "trajectory.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("[llada-traj] wrote %d points to %s", len(records),
                output_dir / "trajectory.jsonl")

    if wandb_log:
        try:
            from utils.logging import log_metrics
            # log the mean final semantic density if tracked
            finals = [r for r in records if r.get("step") == max(p["step"] for p in records)]
            if finals and "semantic_density" in finals[0]:
                log_metrics({"llada_traj/final_semantic_density":
                             sum(r["semantic_density"] for r in finals) / len(finals)})
        except Exception:
            pass

    return {"n_points": len(records)}


def main() -> None:
    parser = argparse.ArgumentParser(description="LLaDA unmasking-step trajectory")
    parser.add_argument("--n-prompts", type=int, default=10)
    parser.add_argument("--set", default="rq5_compositional",
                        help="Prompt set to sample (default: rq5_compositional)")
    parser.add_argument("--track-content", action="store_true",
                        help="Also track compositional content per step (needs Ollama)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--config", nargs="*", metavar="KEY=VALUE")
    args = parser.parse_args()

    from experiments.run_experiment import load_config
    from rewriters.llada_rewriter import LLaDARewriter, LLaDARewriterConfig
    from utils.prompt_io import load_prompts
    import torch

    cfg = load_config(args.config)
    wandb_log = not args.no_wandb

    if wandb_log:
        from utils.logging import init_wandb
        init_wandb(
            project=cfg.get("wandb_project", "prompt-pipeline"),
            run_name="llada_trajectory",
            config={**cfg, "track_content": args.track_content},
            offline=cfg.get("wandb_offline", False),
        )

    llada_raw = cfg.get("llada", {})
    llada_rw = LLaDARewriter(LLaDARewriterConfig(
        model_id=cfg.get("llada_model", "GSAI-ML/LLaDA-8B-Instruct"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        gen_length=llada_raw.get("gen_length", 128),
        steps=llada_raw.get("steps", 128),
        block_length=llada_raw.get("block_length", 32),
        cache_path=None,  # trajectory is a diagnostic, not cached
    ))

    extractor = None
    if args.track_content:
        from evaluation.embedding_analysis import SemanticExtractor
        extractor = SemanticExtractor(
            use_llm=True,
            model=cfg.get("ollama_model", "llama3.1"),
            base_url=cfg.get("ollama_base_url", "http://localhost:11434"),
        )

    backbone = cfg.get("t2i", {}).get("backbone", "sd21")
    output_dir = Path(cfg.get("output_dir", "outputs")) / backbone / "llada_trajectory"

    prompts = load_prompts(args.set)[:args.n_prompts]
    logger.info("LLaDA trajectory | %d prompts from '%s' | track_content=%s",
                len(prompts), args.set, args.track_content)

    run_llada_trajectory(
        prompts=prompts, llada_rewriter=llada_rw, extractor=extractor,
        output_dir=output_dir, wandb_log=wandb_log,
    )

    # Plot immediately.
    try:
        from evaluation.plotting import plot_llada_trajectory
        plot_llada_trajectory(output_dir, output_dir.parent / "plots")
    except Exception as exc:
        logger.warning("Plotting failed (%s)", exc)

    if wandb_log:
        from utils.logging import finish
        finish()
    logger.info("Done.")


if __name__ == "__main__":
    main()

"""
experiments/run_experiment.py

Main entry point for all experiments.

Usage
-----
    python experiments/run_experiment.py --rq 1
    python experiments/run_experiment.py --rq 2 --seed 42 --cfg 7.5
    python experiments/run_experiment.py --rq 3
    python experiments/run_experiment.py --rq 4
    python experiments/run_experiment.py --rq all   # run everything sequentially

    # Dry run (small subset, CPU-safe, no LLaDA):
    python experiments/run_experiment.py --rq 1 --dry-run

    # Override config fields:
    python experiments/run_experiment.py --rq 2 --config default_cfg_scale=5.0

Configuration is loaded from config/experiment.yaml. CLI --config overrides
take precedence over the file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# Add project root to sys.path so modules resolve without installation
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(overrides: list[str] | None = None) -> dict:
    config_path = ROOT / "config" / "experiment.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for override in overrides:
            key, _, val = override.partition("=")
            # Try to parse as Python literal; fall back to string
            try:
                import ast
                cfg[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                cfg[key] = val
    return cfg


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def build_pipelines(cfg: dict, dry_run: bool = False) -> list:
    """
    Construct all ConditioningPipeline instances based on config.

    dry_run: skip LLaDA (requires GPU + 16 GB VRAM) and use CPU encoders.
    """
    from encoders.clip_encoder import CLIPEncoder, CLIPEncoderConfig
    from encoders.longclip_encoder import LongCLIPEncoder, LongCLIPEncoderConfig
    from pipeline.raw import RawPipeline
    from pipeline.ar_rewrite import ARPipeline
    from pipeline.llada_refine import LLaDAPipeline
    from rewriters.ollama_rewriter import OllamaRewriter, OllamaRewriterConfig
    from rewriters.llada_rewriter import LLaDARewriter, LLaDARewriterConfig

    device = "cpu" if dry_run else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.float16

    logger.info("Building pipelines | device=%s | dry_run=%s", device, dry_run)

    # Encoders
    clip_enc = CLIPEncoder(CLIPEncoderConfig(
        model_id=cfg.get("clip_model", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"),
        device=device, torch_dtype=dtype,
    ))
    longclip_enc = LongCLIPEncoder(LongCLIPEncoderConfig(
        checkpoint_path=cfg.get("longclip_checkpoint"),
        device=device, torch_dtype=dtype,
    ))

    encoders = []
    for enc_name in cfg.get("encoders", ["clip"]):
        if enc_name == "clip":
            encoders.append(clip_enc)
        elif enc_name == "longclip":
            encoders.append(longclip_enc)

    pipelines = []
    requested = cfg.get("pipelines", ["raw", "ar", "llada"])

    for enc in encoders:
        if "raw" in requested:
            pipelines.append(RawPipeline(encoder=enc))

        if "ar" in requested:
            cache_dir = cfg.get("rewrite_cache_dir")
            ar_cache = str(Path(cache_dir) / f"ar_{cfg.get('ollama_model', 'llama3.1')}.json") if cache_dir else None
            ollama_cfg = OllamaRewriterConfig(
                model=cfg.get("ollama_model", "llama3.1"),
                base_url=cfg.get("ollama_base_url", "http://localhost:11434"),
                timeout=cfg.get("ollama_timeout", 600),
                cache_path=ar_cache,
            )
            pipelines.append(ARPipeline(encoder=enc, rewriter=OllamaRewriter(ollama_cfg)))

        if "llada" in requested and not dry_run:
            cache_dir = cfg.get("rewrite_cache_dir")
            llada_cache = str(Path(cache_dir) / "llada.json") if cache_dir else None
            llada_cfg_raw = cfg.get("llada", {})
            llada_cfg = LLaDARewriterConfig(
                model_id=cfg.get("llada_model", "GSAI-ML/LLaDA-8B-Instruct"),
                device="cuda" if torch.cuda.is_available() else "cpu",
                gen_length=llada_cfg_raw.get("gen_length", 128),
                steps=llada_cfg_raw.get("steps", 128),
                block_length=llada_cfg_raw.get("block_length", 32),
                temperature=llada_cfg_raw.get("temperature", 0.0),
                cfg_scale=llada_cfg_raw.get("cfg_scale", 0.0),
                cache_path=llada_cache,
            )
            pipelines.append(LLaDAPipeline(encoder=enc, rewriter=LLaDARewriter(llada_cfg)))
        elif "llada" in requested and dry_run:
            logger.info("Skipping LLaDA pipeline in dry-run mode.")

    logger.info("Built %d pipelines: %s", len(pipelines), [p.name for p in pipelines])
    return pipelines


def build_runner(cfg: dict, dry_run: bool = False):
    """Build the T2I image generation runner."""
    from generation.t2i_runner import T2IRunner, T2IRunnerConfig
    device = "cpu" if dry_run else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.float16
    runner_cfg = T2IRunnerConfig(
        model_id=cfg.get("t2i_model", "stabilityai/stable-diffusion-2-1"),
        device=device,
        torch_dtype=dtype,
        num_inference_steps=cfg.get("sampling_steps", 50) if not dry_run else 50,
        default_cfg_scale=cfg.get("default_cfg_scale", 7.5),
        image_height=cfg.get("image_size", 768),
        image_width=cfg.get("image_size", 768),
    )
    return T2IRunner(config=runner_cfg)


def build_scorers(cfg: dict, dry_run: bool = False):
    """
    Build all evaluation scorers.

    In dry-run mode the VQA scorers (BLIP-2, ~15 GB float32 on CPU) and FID
    scorer (Inception-v3) are replaced with stubs that return 0.0 immediately.
    This prevents OOM crashes when testing the pipeline locally on a Mac or
    any machine without enough RAM to load the full scorer stack.
    CLIPScore (~1.5 GB) is still loaded so we get a real signal on prompt–image
    alignment even during a dry run.
    """
    from evaluation.metrics import CLIPScorer, FIDScorer, AttributeBindingScorer, RelationAccuracyScorer
    device = "cpu" if dry_run else ("cuda" if torch.cuda.is_available() else "cpu")
    clip_score_model = cfg.get("clip_score_model", "openai/clip-vit-large-patch14")
    clip_scorer = CLIPScorer(model_name_or_path=clip_score_model, device=device)

    if dry_run:
        logger.info(
            "dry-run: replacing BLIP-2 (VQA) and FID scorers with stubs to avoid OOM. "
            "attr_binding and relation_accuracy will report 0.0."
        )

        class _StubScorer:
            """Returns 0.0 for every call without loading any model."""
            def mean_accuracy(self, images, prompts): return 0.0
            def score(self, image, prompt): return {"accuracy": 0.0, "n_pairs": 0, "n_correct": 0, "details": []}

        class _StubFID:
            def update_real(self, images): pass
            def update_generated(self, images): pass
            def compute(self): return 0.0
            def reset(self): pass

        return clip_scorer, _StubFID(), _StubScorer(), _StubScorer()

    vqa_model = cfg.get("vqa_model", "Salesforce/blip2-flan-t5-xl")
    return (
        clip_scorer,
        FIDScorer(device=device, feature_dims=cfg.get("fid_dims", 2048)),
        AttributeBindingScorer(model_name_or_path=vqa_model, device=device),
        RelationAccuracyScorer(model_name_or_path=vqa_model, device=device),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompt pipeline experiments")
    parser.add_argument("--rq", required=True, choices=["0", "1", "2", "3", "4", "all"],
                        help="Which RQ to run. 0 = warm the rewrite cache only (run before HPC jobs)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed (default: first seed in config)")
    parser.add_argument("--cfg", type=float, default=None,
                        help="Override CFG scale for RQ2/RQ4")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test: small prompt subset, 2 steps, CPU-safe, no LLaDA")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--pipeline", default=None, metavar="NAME",
                        help="Run only the named pipeline (e.g. ar_clip). Used by SLURM array jobs "
                             "to parallelise across pipeline×encoder combinations.")
    parser.add_argument("--config", nargs="*", metavar="KEY=VALUE",
                        help="Override config fields, e.g. --config default_cfg_scale=5.0")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed or cfg["seeds"][0]
    cfg_scale = args.cfg or cfg["default_cfg_scale"]
    wandb_log = not args.no_wandb
    dry_run = args.dry_run
    pipeline_filter = args.pipeline  # e.g. "ar_clip" — None means run all

    from utils.seed import set_seed
    set_seed(seed)

    # W&B init — include pipeline name in run_name so array tasks are distinct
    if wandb_log:
        from utils.logging import init_wandb
        run_suffix = f"_{pipeline_filter}" if pipeline_filter else ""
        init_wandb(
            project=cfg.get("wandb_project", "prompt-pipeline"),
            run_name=f"rq{args.rq}_seed{seed}{run_suffix}{'_dryrun' if dry_run else ''}",
            config={
                **cfg,
                "seed": seed,
                "cfg_scale": cfg_scale,
                "dry_run": dry_run,
                "pipeline_filter": pipeline_filter,
            },
            offline=cfg.get("wandb_offline", False),
        )

    from utils.prompt_io import load_prompts
    output_dir = Path(cfg.get("output_dir", "outputs"))

    # ------------------------------------------------------------------
    # RQ 0 — rewrite cache warm-up (run this before HPC array jobs)
    # Iterates over every prompt used across RQ1-4 and calls each
    # rewriting pipeline's encode_batch() so that all rewrites are saved
    # to disk before any RQ job starts.  Raw pipelines are skipped (no
    # rewriter to cache).  Respects --pipeline to allow parallel warmup
    # (e.g. SLURM array task 0 = ar_clip, task 1 = llada_clip).
    # ------------------------------------------------------------------
    if args.rq == "0":
        from pipeline.raw import RawPipeline

        # Union of every prompt set referenced by any RQ
        all_set_names: set[str] = set()
        for sets in cfg.get("eval_prompt_sets", {}).values():
            all_set_names.update(sets)
        seen: set[str] = set()
        all_prompts: list[str] = []
        for set_name in sorted(all_set_names):
            for p in load_prompts(set_name):
                if p not in seen:
                    seen.add(p)
                    all_prompts.append(p)

        logger.info(
            "Warm-cache mode: %d unique prompts across sets %s",
            len(all_prompts), sorted(all_set_names),
        )

        pipelines = _filter_pipelines(build_pipelines(cfg, dry_run=False))

        for pipeline in pipelines:
            if isinstance(pipeline, RawPipeline):
                logger.info("Skipping %s — no rewriter to cache.", pipeline.name)
                continue
            logger.info(
                "Warming rewrite cache for %s (%d prompts)…",
                pipeline.name, len(all_prompts),
            )
            pipeline.encode_batch(all_prompts)
            logger.info("Cache warm for %s.", pipeline.name)

        logger.info("Rewrite cache complete. Exiting.")
        if wandb_log:
            from utils.logging import finish
            finish()
        return

    # ------------------------------------------------------------------

    rqs_to_run = ["1", "2", "3", "4"] if args.rq == "all" else [args.rq]

    def _filter_pipelines(pipelines: list) -> list:
        """Keep only the requested pipeline when --pipeline is specified."""
        if not pipeline_filter:
            return pipelines
        matched = [p for p in pipelines if p.name == pipeline_filter]
        if not matched:
            available = [p.name for p in pipelines]
            raise ValueError(
                f"--pipeline {pipeline_filter!r} not found. "
                f"Available: {available}"
            )
        logger.info("Pipeline filter active: running only '%s'", pipeline_filter)
        return matched

    for rq in rqs_to_run:
        logger.info("=" * 60)
        logger.info("Running RQ%s%s", rq, f" [{pipeline_filter}]" if pipeline_filter else "")
        logger.info("=" * 60)

        if rq == "1":
            from experiments.rq1_conditioning import run_rq1
            pipelines = _filter_pipelines(build_pipelines(cfg, dry_run))
            prompt_sets = cfg["eval_prompt_sets"]["rq1"]
            prompts = []
            for s in prompt_sets:
                prompts.extend(load_prompts(s))
            if dry_run:
                prompts = prompts[:3]
            run_rq1(pipelines, prompts, output_dir=output_dir / "rq1", wandb_log=wandb_log)

        elif rq == "2":
            from experiments.rq2_compositional import run_rq2
            pipelines = _filter_pipelines(build_pipelines(cfg, dry_run))
            runner = build_runner(cfg, dry_run)
            clip_scorer, fid_scorer, attr_scorer, rel_scorer = build_scorers(cfg, dry_run)
            prompt_sets = {
                s: (load_prompts(s)[:3] if dry_run else load_prompts(s))
                for s in cfg["eval_prompt_sets"]["rq2"]
            }

            print("prompt set", prompt_sets)
            print("pipeline", pipelines)
            run_rq2(
                pipelines=pipelines, runner=runner,
                prompt_sets=prompt_sets,
                clip_scorer=clip_scorer, attr_scorer=attr_scorer, rel_scorer=rel_scorer,
                fid_scorer=fid_scorer,
                seed=seed, cfg_scale=cfg_scale,
                output_dir=output_dir / "rq2", wandb_log=wandb_log,
            )

        elif rq == "3":
            from experiments.rq3_cfg_sensitivity import run_rq3
            pipelines = _filter_pipelines(build_pipelines(cfg, dry_run))
            runner = build_runner(cfg, dry_run)
            clip_scorer, _, attr_scorer, rel_scorer = build_scorers(cfg, dry_run)
            prompts = load_prompts("rq3_sweep")
            if dry_run:
                prompts = prompts[:3]
            cfg_scales = [1.0, 3.0] if dry_run else cfg["cfg_scales"]
            run_rq3(
                pipelines=pipelines, runner=runner, prompts=prompts,
                cfg_scales=cfg_scales,
                clip_scorer=clip_scorer, attr_scorer=attr_scorer, rel_scorer=rel_scorer,
                seed=seed, output_dir=output_dir / "rq3", wandb_log=wandb_log,
            )

        elif rq == "4":
            from experiments.rq4_mechanism import run_rq4
            all_pipelines = _filter_pipelines(build_pipelines(cfg, dry_run))
            runner = build_runner(cfg, dry_run)
            clip_scorer, _, attr_scorer, rel_scorer = build_scorers(cfg, dry_run)

            ar_pipelines = [p for p in all_pipelines if p.name.startswith("ar_")]
            llada_pipelines = [p for p in all_pipelines if p.name.startswith("llada_")]

            prompt_sets = {
                s: (load_prompts(s)[:3] if dry_run else load_prompts(s))
                for s in cfg["eval_prompt_sets"]["rq4"]
            }
            run_rq4(
                ar_pipelines=ar_pipelines, llada_pipelines=llada_pipelines,
                runner=runner, prompt_sets=prompt_sets,
                clip_scorer=clip_scorer, attr_scorer=attr_scorer, rel_scorer=rel_scorer,
                seed=seed, cfg_scale=cfg_scale,
                output_dir=output_dir / "rq4", wandb_log=wandb_log,
            )

    if wandb_log:
        from utils.logging import finish
        finish()

    logger.info("Done.")


if __name__ == "__main__":
    main()

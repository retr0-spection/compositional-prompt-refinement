"""
RQ5 PoE composition ablation (text-only, fast).

Purpose
-------
The initial Product-of-Experts construction gave each expert a SINGLE
constraint ("a red cat"), so experts describing different objects competed for
the same canvas positions and the per-position product collapsed into
comma-salad filler (10/18 prompts). This script compares that original
construction against the "shared_scene" construction, in which every expert
conditions on the WHOLE scene and merely emphasises one constraint, optionally
with a whole-scene fluency-anchor expert.

It is text-only: it calls LLaDARewriter.compose() under each configuration and
scores constraint coverage against the ORIGINAL prompt's scene graph (the same
_coverage used by rq5_text_compare), so a full pass over the 18-prompt set is a
few minutes on one GPU rather than a full image regeneration. Use it to pick a
construction, then run the full RQ5 image pipeline with that config.

Requirements (same as RQ5): a GPU for LLaDA, and Ollama serving the extractor
model for scene-graph decomposition (falls back to keyword extraction if
unavailable, which weakens the constraints).

Usage
-----
    python -m experiments.rq5_poe_ablation \
        --prompt-set rq5_compositional \
        --out outputs/sdxl/rq5_ablation
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field

import torch

from rewriters.llada_rewriter import LLaDARewriter, LLaDARewriterConfig
from evaluation.embedding_analysis import SemanticExtractor
from experiments.rq5_text_compare import _typed_constraints, _coverage
from utils.prompt_io import load_prompts

logger = logging.getLogger(__name__)


@dataclass
class Variant:
    """One PoE construction to evaluate, expressed as config-field overrides."""
    name: str
    expert_mode: str            # "disjoint" | "shared_scene"
    base_expert: bool           # add a whole-scene fluency anchor
    base_weight: float = 1.0
    focus_weight: float = 1.0


# The constructions to compare. "disjoint_nobase" reproduces the original
# (degenerate) behaviour; the shared_scene rows are the proposed fixes.
DEFAULT_VARIANTS: list[Variant] = [
    Variant("disjoint_nobase",  "disjoint",     base_expert=False),
    Variant("shared_nobase",    "shared_scene", base_expert=False),
    Variant("shared_base_w1.0", "shared_scene", base_expert=True, base_weight=1.0),
    Variant("shared_base_w0.5", "shared_scene", base_expert=True, base_weight=0.5),
]


def _is_comma_salad(text: str, min_singletons: int = 6) -> bool:
    """
    Heuristic degeneracy flag: comma-delimited output where many fragments are
    single tokens ("small, round, clear, plastic, cup, with, a, ...").
    """
    frags = [f.strip() for f in text.split(",")]
    singletons = sum(1 for f in frags if 0 < len(f.split()) <= 1)
    return singletons >= min_singletons


def _apply(cfg: LLaDARewriterConfig, v: Variant) -> None:
    cfg.poe_expert_mode = v.expert_mode
    cfg.poe_base_expert = v.base_expert
    cfg.poe_base_weight = v.base_weight
    cfg.poe_focus_weight = v.focus_weight


def run_ablation(
    prompts: list[str],
    rewriter: LLaDARewriter,
    extractor,
    variants: list[Variant],
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cov_keys = ["literal_coverage_all", "literal_coverage_attr",
                "graph_coverage_all", "graph_coverage_attr", "graph_coverage_rel",
                "word_count"]

    per_prompt: list[dict] = []
    for i, prompt in enumerate(prompts):
        constraints, typed = _typed_constraints(prompt, extractor)
        row: dict = {"idx": i, "prompt": prompt, "constraints": constraints,
                     "variants": {}}
        for v in variants:
            _apply(rewriter.config, v)
            text = rewriter.compose(constraints, full_prompt=prompt)
            cov = _coverage(typed, text, extractor)
            row["variants"][v.name] = {
                "text": text,
                "salad": _is_comma_salad(text),
                **{k: cov.get(k) for k in cov_keys},
            }
        per_prompt.append(row)
        logger.info("[ablation] %d/%d done: %r", i + 1, len(prompts), prompt[:48])

    # Aggregate per variant.
    agg: dict = {}
    n = len(per_prompt)
    for v in variants:
        vals = [r["variants"][v.name] for r in per_prompt]
        agg[v.name] = {
            "salad_fraction": sum(1 for x in vals if x["salad"]) / n,
            **{k: sum(x[k] for x in vals) / n for k in cov_keys},
        }

    (out_dir / "ablation.json").write_text(
        json.dumps({"aggregate": agg, "per_prompt": per_prompt}, indent=2))
    _write_report(out_dir / "ablation.txt", per_prompt, agg, variants)
    logger.info("[ablation] wrote %s", out_dir / "ablation.json")
    return agg


def _write_report(path: Path, per_prompt: list[dict], agg: dict,
                  variants: list[Variant]) -> None:
    def f(x) -> str:
        return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("RQ5 PoE construction ablation (text-only)\n")
        fh.write("=" * 60 + "\n\n")
        fh.write("AGGREGATE (mean over prompts)\n")
        head = ("variant", "salad", "lit_all", "gr_all", "gr_attr", "gr_rel", "words")
        fh.write("  {:<18}{:>7}{:>9}{:>8}{:>9}{:>8}{:>7}\n".format(*head))
        for v in variants:
            a = agg[v.name]
            fh.write("  {:<18}{:>7}{:>9}{:>8}{:>9}{:>8}{:>7}\n".format(
                v.name, f(a["salad_fraction"]), f(a["literal_coverage_all"]),
                f(a["graph_coverage_all"]), f(a["graph_coverage_attr"]),
                f(a["graph_coverage_rel"]), f(a["word_count"])))
        fh.write("\n" + "=" * 60 + "\n\nPER-PROMPT OUTPUTS\n\n")
        for r in per_prompt:
            fh.write(f"[{r['idx']}] {r['prompt']}\n")
            fh.write(f"    constraints: {r['constraints']}\n")
            for v in variants:
                x = r["variants"][v.name]
                flag = " [SALAD]" if x["salad"] else ""
                fh.write(f"    {v.name:<18}{flag}\n      {x['text']}\n")
            fh.write("-" * 60 + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-set", default="rq5_compositional")
    ap.add_argument("--out", default="outputs/sdxl/rq5_ablation")
    ap.add_argument("--limit", type=int, default=0, help="cap prompts (0 = all)")
    ap.add_argument("--ollama-model", default="llama3.1")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    args = ap.parse_args()

    prompts = load_prompts(args.prompt_set)
    if args.limit:
        prompts = prompts[:args.limit]

    rewriter = LLaDARewriter(LLaDARewriterConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
    ))
    extractor = SemanticExtractor(
        use_llm=True, model=args.ollama_model, base_url=args.ollama_url)

    agg = run_ablation(prompts, rewriter, extractor, DEFAULT_VARIANTS,
                       Path(args.out))
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()

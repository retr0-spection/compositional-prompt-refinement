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
_coverage used by rq5_text_compare). No image generation.

Robustness
----------
Results are written incrementally (one JSON line per prompt) and the aggregate
report is refreshed after every prompt, so a wall-clock timeout still leaves
partial results on disk. Re-running resumes: prompts already in ablation.jsonl
are skipped. Per-compose timing is logged to stdout so a slow run is visible.

Requirements (same as RQ5): a GPU for LLaDA, and Ollama serving the extractor
model for scene-graph decomposition (falls back to keyword extraction if
unavailable, which weakens the constraints).

Usage
-----
    python -m experiments.rq5_poe_ablation --out outputs/sdxl/rq5_ablation
    # fast first pass: fewer prompts, fewer denoising steps
    python -m experiments.rq5_poe_ablation --limit 3 --steps 64 \
        --variants disjoint_nobase,shared_base_w1.0
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass

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
ALL_VARIANTS: dict[str, Variant] = {
    v.name: v for v in [
        Variant("disjoint_nobase",  "disjoint",     base_expert=False),
        Variant("shared_nobase",    "shared_scene", base_expert=False),
        Variant("shared_base_w1.0", "shared_scene", base_expert=True, base_weight=1.0),
        Variant("shared_base_w0.5", "shared_scene", base_expert=True, base_weight=0.5),
    ]
}
DEFAULT_ORDER = list(ALL_VARIANTS.keys())

COV_KEYS = ["literal_coverage_all", "literal_coverage_attr",
            "graph_coverage_all", "graph_coverage_attr", "graph_coverage_rel",
            "word_count"]


def _is_comma_salad(text: str, min_singletons: int = 6) -> bool:
    """Comma-delimited output where many fragments are single tokens."""
    frags = [f.strip() for f in text.split(",")]
    singletons = sum(1 for f in frags if 0 < len(f.split()) <= 1)
    return singletons >= min_singletons


def _apply(cfg: LLaDARewriterConfig, v: Variant) -> None:
    cfg.poe_expert_mode = v.expert_mode
    cfg.poe_base_expert = v.base_expert
    cfg.poe_base_weight = v.base_weight
    cfg.poe_focus_weight = v.focus_weight


def _aggregate(rows: list[dict], variants: list[Variant]) -> dict:
    agg: dict = {}
    n = max(len(rows), 1)
    for v in variants:
        vals = [r["variants"][v.name] for r in rows if v.name in r["variants"]]
        m = max(len(vals), 1)
        agg[v.name] = {
            "n": len(vals),
            "salad_fraction": sum(1 for x in vals if x["salad"]) / m,
            **{k: sum(x[k] for x in vals) / m for k in COV_KEYS},
        }
    return agg


def _write_report(path: Path, rows: list[dict], variants: list[Variant]) -> None:
    agg = _aggregate(rows, variants)
    def f(x) -> str:
        return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("RQ5 PoE construction ablation (text-only)\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"AGGREGATE (mean over {len(rows)} prompts so far)\n")
        head = ("variant", "n", "salad", "lit_all", "gr_all", "gr_attr", "gr_rel", "words")
        fh.write("  {:<18}{:>4}{:>7}{:>9}{:>8}{:>9}{:>8}{:>7}\n".format(*head))
        for v in variants:
            a = agg[v.name]
            fh.write("  {:<18}{:>4}{:>7}{:>9}{:>8}{:>9}{:>8}{:>7}\n".format(
                v.name, a["n"], f(a["salad_fraction"]), f(a["literal_coverage_all"]),
                f(a["graph_coverage_all"]), f(a["graph_coverage_attr"]),
                f(a["graph_coverage_rel"]), f(a["word_count"])))
        fh.write("\n" + "=" * 60 + "\n\nPER-PROMPT OUTPUTS\n\n")
        for r in rows:
            fh.write(f"[{r['idx']}] {r['prompt']}\n")
            fh.write(f"    constraints: {r['constraints']}\n")
            for v in variants:
                x = r["variants"].get(v.name)
                if not x:
                    continue
                flag = " [SALAD]" if x["salad"] else ""
                fh.write(f"    {v.name:<18}{flag}\n      {x['text']}\n")
            fh.write("-" * 60 + "\n")
    # machine-readable aggregate alongside the readable report
    path.with_name("ablation.json").write_text(
        json.dumps({"aggregate": agg, "n_prompts": len(rows)}, indent=2))


def run_ablation(prompts: list[str], rewriter: LLaDARewriter, extractor,
                variants: list[Variant], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "ablation.jsonl"

    # Resume: load any prompts already completed.
    rows: list[dict] = []
    done: set[int] = set()
    if jsonl.exists():
        for line in jsonl.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append(r)
            done.add(r["idx"])
        if done:
            logger.info("[ablation] resuming: %d prompts already done", len(done))

    fh = jsonl.open("a", encoding="utf-8")
    for i, prompt in enumerate(prompts):
        if i in done:
            continue
        t_prompt = time.perf_counter()
        constraints, typed = _typed_constraints(prompt, extractor)
        row: dict = {"idx": i, "prompt": prompt, "constraints": constraints,
                     "variants": {}}
        for v in variants:
            _apply(rewriter.config, v)
            t0 = time.perf_counter()
            text = rewriter.compose(constraints, full_prompt=prompt)
            dt = time.perf_counter() - t0
            cov = _coverage(typed, text, extractor)
            row["variants"][v.name] = {
                "text": text,
                "salad": _is_comma_salad(text),
                "compose_seconds": round(dt, 2),
                **{k: cov.get(k) for k in COV_KEYS},
            }
            logger.info("[ablation] prompt %d/%d | %-18s | %.1fs | salad=%s",
                        i + 1, len(prompts), v.name, dt,
                        row["variants"][v.name]["salad"])
        rows.append(row)
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        _write_report(out_dir / "ablation.txt", rows, variants)  # refresh each prompt
        logger.info("[ablation] prompt %d/%d DONE in %.1fs (results flushed)",
                    i + 1, len(prompts), time.perf_counter() - t_prompt)
    fh.close()

    agg = _aggregate(rows, variants)
    logger.info("[ablation] complete: %d prompts", len(rows))
    return agg


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-set", default="rq5_compositional")
    ap.add_argument("--out", default="outputs/sdxl/rq5_ablation")
    ap.add_argument("--limit", type=int, default=0, help="cap prompts (0 = all)")
    ap.add_argument("--steps", type=int, default=0,
                    help="override LLaDA denoising steps (0 = config default 128; "
                         "use 64 for a faster first pass)")
    ap.add_argument("--variants", default="",
                    help="comma-separated subset of: " + ",".join(DEFAULT_ORDER))
    ap.add_argument("--ollama-model", default="llama3.1")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    args = ap.parse_args()

    names = [s for s in args.variants.split(",") if s] or DEFAULT_ORDER
    variants = [ALL_VARIANTS[n] for n in names]

    prompts = load_prompts(args.prompt_set)
    if args.limit:
        prompts = prompts[:args.limit]

    logger.info("[ablation] %d prompts | variants=%s | steps=%s",
                len(prompts), names, args.steps or "default")

    t_load = time.perf_counter()
    rewriter = LLaDARewriter(LLaDARewriterConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
    ))
    if args.steps:
        rewriter.config.steps = args.steps
    rewriter._load()  # surface model-load time up front rather than mid-loop
    logger.info("[ablation] LLaDA loaded in %.1fs on %s",
                time.perf_counter() - t_load, rewriter.config.device)

    extractor = SemanticExtractor(
        use_llm=True, model=args.ollama_model, base_url=args.ollama_url)

    agg = run_ablation(prompts, rewriter, extractor, variants, Path(args.out))
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()

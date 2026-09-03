"""
experiments/rq5_poe.py

RQ5 (capstone): Does language-side Product-of-Experts composition, a
capability unique to the diffusion LM, improve compositional generation over
single-pass refinement?

The claim: an autoregressive model expands a prompt in one left-to-right pass;
it cannot hold N constraints in superposition and resolve them jointly. LLaDA's
masked diffusion CAN — PoE conditions one shared response canvas on multiple
constraint "experts" and unmasks only where they jointly agree (see
rewriters/llada_rewriter.py :: _generate_poe / compose).

Four conditions per compositional prompt, same seed / backbone / encoder:
    raw          — the compositional prompt fed directly (floor)
    ar           — Ollama expands the whole prompt in one pass
    llada_single — LLaDA rewrite() on the whole prompt (isolates PoE from LLaDA)
    llada_poe    — LLaDA compose() over decomposed constraints (the capstone)

The decisive comparison is llada_poe vs llada_single: same model, same
mechanism, differing ONLY in whether constraints are composed jointly. If PoE
wins there, the composition operation itself helps, independent of the model.

Decomposition: constraints come from the scene-graph extractor
(evaluation.embedding_analysis.SemanticExtractor) — its [obj, attr] bindings
and [o1, rel, o2] triples ARE the constraints, tying RQ5 to the RQ1 machinery.

Proof-of-concept scale: a small set of hard compositional prompts (~15-20).
Holds the encoder fixed (whatever the backbone provides).

Output: outputs/<backbone>/rq5/<condition>/... images + trace.jsonl, plus
rq5_comparison.json with PoE-vs-baseline deltas.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

_CONDITIONS = ["raw", "ar", "llada_single", "llada_poe"]


def _decompose(prompt: str, extractor) -> list[str]:
    """
    Turn a compositional prompt into a list of atomic constraint strings using
    the scene-graph extractor.

    "a red cat left of a blue dog"
      -> ["a red cat", "a blue dog", "cat left of dog"]

    Falls back to the whole prompt as a single constraint if extraction yields
    nothing usable (so PoE still runs, just with one expert = degenerate case).
    """
    graph = extractor.extract(prompt)
    constraints: list[str] = []

    # Attribute bindings: [obj, attr] -> "a attr obj"
    for pair in graph.get("attributes", []):
        if len(pair) == 2 and pair[0] != "?" and pair[1] != "?":
            constraints.append(f"a {pair[1]} {pair[0]}")

    # Relation triples: [o1, rel, o2] -> "o1 rel o2"
    for tri in graph.get("relations", []):
        if len(tri) == 3 and "?" not in tri:
            constraints.append(f"{tri[0]} {tri[1]} {tri[2]}")

    # If we got objects but no bindings/relations, at least constrain objects.
    if not constraints:
        objs = [o for o in graph.get("objects", []) if o and o != "?"]
        constraints = [f"a {o}" for o in objs]

    # Degenerate fallback: whole prompt as one constraint.
    if not constraints:
        constraints = [prompt]

    return constraints


def _generate_images(runner, texts: list[str], cfg_scale: float, seed: int,
                     img_dir: Path) -> list[Image.Image]:
    """Generate (or resume from disk) one image per text. Backbone-agnostic."""
    from utils.naming import img_name
    img_dir.mkdir(parents=True, exist_ok=True)

    paths = [img_dir / img_name(i, cfg_scale, seed) for i in range(len(texts))]
    images: list[Optional[Image.Image]] = [None] * len(texts)
    missing = []
    for i, p in enumerate(paths):
        if p.exists():
            try:
                images[i] = Image.open(p).convert("RGB")
            except Exception:
                missing.append(i)
        else:
            missing.append(i)

    if missing:
        logger.info("[RQ5] generating %d/%d images in %s", len(missing), len(texts), img_dir)
        if hasattr(runner, "_encode"):  # T2IRunnerV2 (text-in)
            new = runner.generate_batch(
                prompts=[texts[i] for i in missing],
                cfg_scale=cfg_scale, seeds=[seed] * len(missing),
            )
        else:  # legacy embedding-in runner — RQ5 requires v2; guard
            raise RuntimeError("RQ5 requires the text-in T2IRunnerV2 (set a t2i: block).")
        for i, img in zip(missing, new):
            img.save(paths[i])
            images[i] = img
    return images  # type: ignore


def run_rq5(
    prompts: list[str],
    ar_rewriter,
    llada_rewriter,
    extractor,
    runner,
    clip_scorer,
    attr_scorer,
    rel_scorer,
    seed: int = 42,
    cfg_scale: float = 7.5,
    output_dir: str | Path = "outputs/rq5",
    wandb_log: bool = True,
) -> dict:
    """
    Run the four-condition PoE comparison over a small compositional prompt set.

    Returns a dict of per-condition aggregates plus PoE-vs-baseline deltas,
    also written to rq5_comparison.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: TEXT-level comparison (before any image) ----
    # Runs first so we can see whether compose() even produces different text
    # from rewrite(), with constraint-coverage broken down by type. Returns the
    # per-prompt records (incl. the single_text and poe_text) so the image
    # stage below REUSES them rather than recomputing compose() (which is
    # expensive and uncached).
    from experiments.rq5_text_compare import run_text_compare, _typed_constraints
    logger.info("[RQ5] Stage 1: text-level comparison (single vs PoE)")
    text_agg, text_records = run_text_compare(
        prompts=prompts, llada_rewriter=llada_rewriter, extractor=extractor,
        output_dir=output_dir.parent / "rq5_text", wandb_log=wandb_log,
    )
    # index text records by prompt idx for reuse
    text_by_idx = {r["idx"]: r for r in text_records}

    # ---- Stage 2: build the conditioning TEXT for each condition ----
    logger.info("[RQ5] Stage 2: preparing conditioning text for %d prompts x 4 conditions", len(prompts))
    texts: dict[str, list[str]] = {c: [] for c in _CONDITIONS}
    decompositions: list[list[str]] = []
    typed_per_prompt: list[list[dict]] = []

    for i, prompt in enumerate(prompts):
        rec = text_by_idx.get(i, {})
        constraints = rec.get("constraints") or _typed_constraints(prompt, extractor)[0]
        typed = rec.get("_typed") or _typed_constraints(prompt, extractor)[1]
        decompositions.append(constraints)
        typed_per_prompt.append(typed)

        texts["raw"].append(prompt)
        texts["ar"].append(ar_rewriter.rewrite(prompt))
        # Reuse the rewrites already computed in Stage 1 (avoid recomputing the
        # uncached compose()); fall back to computing if absent.
        texts["llada_single"].append(
            rec.get("single_text") or llada_rewriter.rewrite(prompt))
        texts["llada_poe"].append(
            rec.get("poe_text") or llada_rewriter.compose(constraints))
        logger.info("[RQ5] %r -> %d constraints", prompt[:50], len(constraints))

    # ---- Generate + score each condition ----
    from utils.logging import log_metrics
    results: dict[str, list[dict]] = {c: [] for c in _CONDITIONS}

    for cond in _CONDITIONS:
        img_dir = output_dir / cond
        images = _generate_images(runner, texts[cond], cfg_scale, seed, img_dir)

        for i, (img, prompt) in enumerate(zip(images, prompts)):
            # Score against the ORIGINAL compositional prompt (user intent),
            # not the refined text — we want compositional fidelity to intent.
            clip_s = clip_scorer.score(img, prompt)
            attr_r = attr_scorer.score(img, prompt)
            rel_r = rel_scorer.score(img, prompt)
            results[cond].append({
                "idx": i,
                "condition": cond,
                "prompt": prompt,
                "conditioning_text": texts[cond][i],
                "constraints": decompositions[i],
                "image_path": str(img_dir / f"prompt_{i:03d}_cfg{cfg_scale:g}_seed{seed}.png"),
                "clip_score": clip_s,
                "attr_binding": attr_r,
                "relation_accuracy": rel_r,
            })

        with open(output_dir / cond / "trace.jsonl", "w", encoding="utf-8") as f:
            for rec in results[cond]:
                f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")

    # ---- Aggregate (applicable-prompt means) ----
    def _agg(cond: str) -> dict:
        recs = results[cond]
        n = len(recs) or 1
        attr = [r["attr_binding"]["accuracy"] for r in recs
                if r["attr_binding"].get("n_pairs", 0) > 0]
        rel = [r["relation_accuracy"]["accuracy"] for r in recs
               if r["relation_accuracy"].get("n_relations", 0) > 0]
        return {
            "clip_score": sum(r["clip_score"] for r in recs) / n,
            "attr_binding_accuracy": sum(attr) / len(attr) if attr else float("nan"),
            "relation_accuracy": sum(rel) / len(rel) if rel else float("nan"),
            "n_attr_scored": len(attr),
            "n_rel_scored": len(rel),
        }

    agg = {c: _agg(c) for c in _CONDITIONS}

    # ---- Per-constraint-TYPE aggregation (mean-field signature test) ----
    # Attribute binding is a per-position property; spatial relations are a
    # cross-position property. The mean-field PoE approximation is expected to
    # help attributes more than relations. Reporting attr-vs-rel deltas
    # separately surfaces that signature on the IMAGE side (mirroring the
    # text-coverage per-type split in Stage 1).
    #   attr_binding_accuracy already isolates attributes.
    #   relation_accuracy already isolates relations.
    # So the per-type deltas are just those two metrics compared PoE vs single.
    per_type = {
        "attr": {
            "single": agg["llada_single"]["attr_binding_accuracy"],
            "poe": agg["llada_poe"]["attr_binding_accuracy"],
        },
        "rel": {
            "single": agg["llada_single"]["relation_accuracy"],
            "poe": agg["llada_poe"]["relation_accuracy"],
        },
    }
    for t in per_type:
        s, p = per_type[t]["single"], per_type[t]["poe"]
        per_type[t]["delta_poe_minus_single"] = (
            p - s if (s == s and p == p) else float("nan"))

    # ---- PoE vs each baseline (the decisive comparisons) ----
    def _delta(metric: str, baseline: str) -> float:
        a, b = agg["llada_poe"][metric], agg[baseline][metric]
        if a != a or b != b:  # NaN
            return float("nan")
        return a - b

    comparison = {}
    for baseline in ["raw", "ar", "llada_single"]:
        for metric in ["clip_score", "attr_binding_accuracy", "relation_accuracy"]:
            comparison[f"poe_vs_{baseline}/{metric}"] = _delta(metric, baseline)

    # ---- Persist ----
    with open(output_dir / "rq5_aggregate.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, default=str)
    with open(output_dir / "rq5_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=str)
    with open(output_dir / "rq5_per_type.json", "w", encoding="utf-8") as f:
        json.dump(per_type, f, indent=2, default=str)

    with open(output_dir / "rq5_summary.txt", "w", encoding="utf-8") as f:
        f.write("RQ5 - Product-of-Experts Composition (capstone)\n")
        f.write("=" * 50 + "\n\n")
        for cond in _CONDITIONS:
            a = agg[cond]
            f.write(f"Condition: {cond}\n")
            f.write(f"  CLIPScore:      {a['clip_score']:.4f}\n")
            attr = f"{a['attr_binding_accuracy']:.4f}" if a['attr_binding_accuracy'] == a['attr_binding_accuracy'] else "n/a"
            rel = f"{a['relation_accuracy']:.4f}" if a['relation_accuracy'] == a['relation_accuracy'] else "n/a"
            f.write(f"  Attr binding:   {attr} (n={a['n_attr_scored']})\n")
            f.write(f"  Relation acc:   {rel} (n={a['n_rel_scored']})\n\n")
        f.write("PoE vs baselines (positive = PoE wins):\n")
        for k, v in sorted(comparison.items()):
            vs = f"{v:+.4f}" if v == v else "n/a"
            f.write(f"  {k}: {vs}\n")
        f.write("\nPer-constraint-type (PoE vs single) - mean-field signature:\n")
        for t in ("attr", "rel"):
            d = per_type[t]["delta_poe_minus_single"]
            ds = f"{d:+.4f}" if d == d else "n/a"
            f.write(f"  {t}: single={per_type[t]['single']}, "
                    f"poe={per_type[t]['poe']}, delta={ds}\n")

    if wandb_log:
        flat = {}
        for cond in _CONDITIONS:
            for m, val in agg[cond].items():
                flat[f"rq5/{cond}/{m}"] = val
        for k, v in comparison.items():
            flat[f"rq5/{k}"] = v
        log_metrics(flat)

    logger.info("[RQ5] Done. Key comparison (PoE vs llada_single): %s",
                {m: round(comparison[f"poe_vs_llada_single/{m}"], 4)
                 for m in ["clip_score", "attr_binding_accuracy", "relation_accuracy"]
                 if comparison[f"poe_vs_llada_single/{m}"] == comparison[f"poe_vs_llada_single/{m}"]})
    return {"aggregate": agg, "comparison": comparison, "per_type": per_type,
            "text_comparison": text_agg}

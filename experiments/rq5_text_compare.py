"""
experiments/rq5_text_compare.py

RQ5 text-level comparison: does language-side Product-of-Experts composition
produce DIFFERENT (and better-covering) rewrite TEXT than single-pass LLaDA,
BEFORE any image is generated?

This isolates the language layer. For each compositional prompt we decompose it
into N constraints, then produce two rewrites:

    llada_single = LLaDARewriter.rewrite(prompt)        # one prefix, one pass
    llada_poe    = LLaDARewriter.compose(constraints)   # N experts, product

and compare the two output STRINGS on:

  1. Constraint coverage (two independent measures):
       - graph_coverage : fraction of constraints whose objects/attributes/
                          relations reappear in the output's OWN scene graph
                          (extractor-checked, robust to paraphrase)
       - literal_coverage: fraction of constraints whose surface tokens (object
                          + bound attribute; object1 + relation + object2) are
                          literally present in the output text
     Both are reported AGGREGATE and split by TYPE (attributes vs relations),
     because the mean-field PoE approximation is expected to help attributes
     (per-position) more than relations (cross-position) — a difference in
     per-type coverage is the signature of that.

  2. Compositional density: attribute + relation counts (scene-graph) per output.
  3. Length: word count.

It also DUMPS the raw (single, poe) text pair per prompt for qualitative
reading, auto-flagging the prompts where the two diverge most.

No images, no T2I backbone — only LLaDA + the Ollama-backed extractor. This is
the cheapest test of whether compose() does anything different from rewrite().

Output (outputs/<backbone>/rq5_text/):
    text_pairs.jsonl      one record per prompt: constraints, both texts, metrics
    text_compare.json     aggregate metrics (single vs poe)
    text_compare.txt      human-readable table + flagged divergences
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constraint decomposition + typed constraint records
# ---------------------------------------------------------------------------

def _typed_constraints(prompt: str, extractor) -> tuple[list[str], list[dict]]:
    """
    Decompose a prompt into constraint strings AND typed constraint records.

    Returns (constraint_strings, typed) where typed is a list of dicts:
        {"type": "attribute", "obj": "cat", "attr": "red", "text": "a red cat"}
        {"type": "relation", "o1": "cat", "rel": "left of", "o2": "dog",
         "text": "cat left of dog"}
    The strings feed compose(); the typed records drive coverage checks.
    """
    graph = extractor.extract(prompt)
    strings: list[str] = []
    typed: list[dict] = []

    for pair in graph.get("attributes", []):
        if len(pair) == 2 and pair[0] != "?" and pair[1] != "?":
            obj, attr = str(pair[0]), str(pair[1])
            s = f"a {attr} {obj}"
            strings.append(s)
            typed.append({"type": "attribute", "obj": obj, "attr": attr, "text": s})

    for tri in graph.get("relations", []):
        if len(tri) == 3 and "?" not in tri:
            o1, rel, o2 = str(tri[0]), str(tri[1]), str(tri[2])
            s = f"{o1} {rel} {o2}"
            strings.append(s)
            typed.append({"type": "relation", "o1": o1, "rel": rel, "o2": o2, "text": s})

    if not strings:
        objs = [o for o in graph.get("objects", []) if o and o != "?"]
        for o in objs:
            strings.append(f"a {o}")
            typed.append({"type": "object", "obj": str(o), "text": f"a {o}"})

    if not strings:
        strings = [prompt]
        typed = [{"type": "raw", "text": prompt}]

    return strings, typed


# ---------------------------------------------------------------------------
# Coverage measures
# ---------------------------------------------------------------------------

def _literal_covered(constraint: dict, text: str) -> bool:
    """Surface-token containment: are the constraint's key words in the text?"""
    t = text.lower()
    if constraint["type"] == "attribute":
        return constraint["obj"].lower() in t and constraint["attr"].lower() in t
    if constraint["type"] == "relation":
        rel_head = constraint["rel"].lower().split()[0]  # 'left of' -> 'left'
        return (constraint["o1"].lower() in t and constraint["o2"].lower() in t
                and rel_head in t)
    if constraint["type"] == "object":
        return constraint["obj"].lower() in t
    return constraint["text"].lower() in t


def _graph_covered(constraint: dict, out_graph: dict) -> bool:
    """
    Scene-graph containment: does the output's OWN extracted graph contain this
    constraint (robust to paraphrase). Checked structurally: object present,
    and (attributes) some attribute bound to it; (relations) some relation
    between the two objects.
    """
    objs = {str(o).lower() for o in out_graph.get("objects", [])}
    attrs = [(str(a[0]).lower(), str(a[1]).lower())
             for a in out_graph.get("attributes", []) if len(a) == 2]
    rels = [(str(r[0]).lower(), str(r[2]).lower())
            for r in out_graph.get("relations", []) if len(r) == 3]

    if constraint["type"] == "attribute":
        o = constraint["obj"].lower()
        return any(ao == o for ao, _ in attrs) or o in objs
    if constraint["type"] == "relation":
        o1, o2 = constraint["o1"].lower(), constraint["o2"].lower()
        return any({r1, r2} == {o1, o2} for r1, r2 in rels)
    if constraint["type"] == "object":
        return constraint["obj"].lower() in objs
    return False


def _coverage(typed: list[dict], text: str, extractor) -> dict:
    """Literal + graph coverage, aggregate and per-type, for one output."""
    out_graph = extractor.extract(text)

    def _frac(pred, items):
        items = list(items)
        return (sum(1 for c in items if pred(c)) / len(items)) if items else float("nan")

    attrs = [c for c in typed if c["type"] == "attribute"]
    rels = [c for c in typed if c["type"] == "relation"]

    return {
        "literal_coverage_all": _frac(lambda c: _literal_covered(c, text), typed),
        "literal_coverage_attr": _frac(lambda c: _literal_covered(c, text), attrs),
        "literal_coverage_rel": _frac(lambda c: _literal_covered(c, text), rels),
        "graph_coverage_all": _frac(lambda c: _graph_covered(c, out_graph), typed),
        "graph_coverage_attr": _frac(lambda c: _graph_covered(c, out_graph), attrs),
        "graph_coverage_rel": _frac(lambda c: _graph_covered(c, out_graph), rels),
        "attr_count": out_graph.get("n_attribute_tokens", len(out_graph.get("attributes", []))),
        "rel_count": out_graph.get("n_relation_tokens", len(out_graph.get("relations", []))),
        "word_count": len(text.split()),
    }


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_text_compare(
    prompts: list[str],
    llada_rewriter,
    extractor,
    output_dir: str | Path = "outputs/rq5_text",
    wandb_log: bool = True,
) -> tuple[dict, list[dict]]:
    """
    Compare rewrite() vs compose() TEXT for each compositional prompt.

    Returns (aggregate_metrics, per_prompt_records). The records include
    single_text, poe_text and the typed constraints, so the caller (RQ5 image
    stage) can REUSE them instead of recomputing the uncached compose().
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for i, prompt in enumerate(prompts):
        constraints, typed = _typed_constraints(prompt, extractor)
        logger.info("[rq5-text] %d/%d: %r -> %d constraints",
                    i + 1, len(prompts), prompt[:50], len(constraints))

        single_text = llada_rewriter.rewrite(prompt)
        poe_text = llada_rewriter.compose(constraints)

        single_cov = _coverage(typed, single_text, extractor)
        poe_cov = _coverage(typed, poe_text, extractor)

        s_tok, p_tok = set(single_text.lower().split()), set(poe_text.lower().split())
        divergence = (len(s_tok ^ p_tok) / len(s_tok | p_tok)) if (s_tok | p_tok) else 0.0

        records.append({
            "idx": i,
            "prompt": prompt,
            "n_constraints": len(constraints),
            "constraints": [c["text"] for c in typed],
            "constraint_types": [c["type"] for c in typed],
            "_typed": typed,   # full typed records, for reuse by the image stage
            "single_text": single_text,
            "poe_text": poe_text,
            "divergence": divergence,
            "single": single_cov,
            "poe": poe_cov,
        })

    # persist per-prompt records (drop _typed from disk to keep it readable)
    with open(output_dir / "text_pairs.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            disk = {k: v for k, v in r.items() if k != "_typed"}
            f.write(json.dumps(disk, ensure_ascii=False) + "\n")

    def _mean(cond, key):
        vals = [r[cond][key] for r in records
                if r[cond].get(key) is not None and r[cond][key] == r[cond][key]]
        return sum(vals) / len(vals) if vals else float("nan")

    metrics = ["literal_coverage_all", "literal_coverage_attr", "literal_coverage_rel",
               "graph_coverage_all", "graph_coverage_attr", "graph_coverage_rel",
               "attr_count", "rel_count", "word_count"]
    agg = {
        "single": {m: _mean("single", m) for m in metrics},
        "poe": {m: _mean("poe", m) for m in metrics},
        "mean_divergence": (sum(r["divergence"] for r in records) / len(records)
                            if records else 0.0),
    }
    agg["delta_poe_minus_single"] = {
        m: (agg["poe"][m] - agg["single"][m]
            if (agg["poe"][m] == agg["poe"][m] and agg["single"][m] == agg["single"][m])
            else float("nan"))
        for m in metrics
    }

    with open(output_dir / "text_compare.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, default=str)

    _write_report(output_dir / "text_compare.txt", records, agg)

    if wandb_log:
        try:
            from utils.logging import log_metrics
            flat = {}
            for cond in ("single", "poe"):
                for m in metrics:
                    flat[f"rq5_text/{cond}/{m}"] = agg[cond][m]
            for m in metrics:
                flat[f"rq5_text/delta/{m}"] = agg["delta_poe_minus_single"][m]
            flat["rq5_text/mean_divergence"] = agg["mean_divergence"]
            log_metrics(flat)
        except Exception:
            pass

    logger.info("[rq5-text] done. graph-coverage delta (poe-single) all=%.3f attr=%.3f rel=%.3f",
                agg["delta_poe_minus_single"]["graph_coverage_all"],
                agg["delta_poe_minus_single"]["graph_coverage_attr"],
                agg["delta_poe_minus_single"]["graph_coverage_rel"])
    return agg, records


def _write_report(path: Path, records: list[dict], agg: dict) -> None:
    def _f(x):
        return f"{x:.3f}" if isinstance(x, float) and x == x else "n/a"

    with open(path, "w", encoding="utf-8") as f:
        f.write("RQ5 TEXT COMPARISON - LLaDA (single) vs LLaDA+PoE (compose)\n")
        f.write("=" * 62 + "\n\n")

        f.write("AGGREGATE METRICS (mean over prompts)\n")
        f.write("-" * 62 + "\n")
        f.write(f"{'metric':32s} {'single':>9s} {'poe':>9s} {'d(poe-single)':>14s}\n")
        for m in ["graph_coverage_all", "graph_coverage_attr", "graph_coverage_rel",
                  "literal_coverage_all", "literal_coverage_attr", "literal_coverage_rel",
                  "attr_count", "rel_count", "word_count"]:
            f.write(f"{m:32s} {_f(agg['single'][m]):>9s} {_f(agg['poe'][m]):>9s} "
                    f"{_f(agg['delta_poe_minus_single'][m]):>14s}\n")
        f.write(f"\nmean token-set divergence (single vs poe): "
                f"{agg['mean_divergence']:.3f}\n\n")

        flagged = sorted(records, key=lambda r: r["divergence"], reverse=True)[:5]
        f.write("TOP DIVERGENCES (single vs poe differ most)\n")
        f.write("-" * 62 + "\n")
        for r in flagged:
            f.write(f"\n[{r['idx']}] {r['prompt']}   (divergence {r['divergence']:.2f})\n")
            f.write(f"  constraints: {r['constraints']}\n")
            f.write(f"  SINGLE: {r['single_text']}\n")
            f.write(f"  POE   : {r['poe_text']}\n")

        f.write("\n\nALL TEXT PAIRS\n")
        f.write("=" * 62 + "\n")
        for r in records:
            f.write(f"\n[{r['idx']}] {r['prompt']}\n")
            f.write(f"  constraints ({r['n_constraints']}): {r['constraints']}\n")
            f.write(f"  SINGLE: {r['single_text']}\n")
            f.write(f"  POE   : {r['poe_text']}\n")
            f.write(f"  cov(graph all/attr/rel)  single: "
                    f"{_f(r['single']['graph_coverage_all'])}/"
                    f"{_f(r['single']['graph_coverage_attr'])}/"
                    f"{_f(r['single']['graph_coverage_rel'])}   poe: "
                    f"{_f(r['poe']['graph_coverage_all'])}/"
                    f"{_f(r['poe']['graph_coverage_attr'])}/"
                    f"{_f(r['poe']['graph_coverage_rel'])}\n")

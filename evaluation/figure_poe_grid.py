"""
evaluation/figure_poe_grid.py

Supplementary dissertation/paper figure: LLaDA vs LLaDA+PoE, paired rows.

Layout (paper-style, "3xN" blocks):
    Each block is 2 rows x C columns (default C=3).
        top row    : LLaDA (single-pass rewrite)   condition = llada_single
        bottom row : LLaDA + PoE (joint composition) condition = llada_poe
    The same prompt sits in the same column top and bottom, so a reader compares
    single-pass vs composed for one prompt by looking down the column.
    For N examples with C columns, ceil(N/C) blocks are stacked down the page.

Example selection: auto-picks the prompts where PoE beats LLaDA_single by the
largest margin on the decisive metric (attribute binding, then relation, then
CLIPScore), i.e. the clearest "composition helps" cases. Use --select to change.

Reads RQ5 outputs already on disk:
    outputs/<backbone>/rq5/llada_single/   images + trace.jsonl
    outputs/<backbone>/rq5/llada_poe/      images + trace.jsonl

Output: outputs/plots/figures/poe_grid.{png,pdf} with a caption rendered under
the grid (paper-ready) and, alongside, poe_grid_caption.txt for the LaTeX
\\caption{} body.

Usage:
    python -m evaluation.figure_poe_grid --backbone sdxl --n 6
    python -m evaluation.figure_poe_grid --backbone sd21 --n 9 --cols 3
    python -m evaluation.figure_poe_grid --backbone sdxl --indices 0 3 4 7 --cols 2
    python -m evaluation.figure_poe_grid --backbone sdxl --select first --n 6
    python -m evaluation.figure_poe_grid --backbone sdxl --columns   # side-by-side variant
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_TOP_LABEL = "LLaDA"
_BOTTOM_LABEL = "LLaDA + PoE"


# ---------------------------------------------------------------------------
# Trace / score helpers
# ---------------------------------------------------------------------------

def _load_trace(trace_path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if not trace_path.exists():
        return out
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            out[r.get("idx", len(out))] = r
        except json.JSONDecodeError:
            continue
    return out


def _metric(rec: Optional[dict]) -> tuple[float, str]:
    """
    Return (scalar, label) for ranking/annotation. Prefers attribute binding,
    then relation, then CLIPScore, so the 'decisive' compositional metric drives
    selection where it applies.
    """
    if not rec:
        return float("nan"), ""
    ab = rec.get("attr_binding")
    if isinstance(ab, dict) and ab.get("n_pairs", 0) > 0:
        return ab.get("accuracy", float("nan")), "attr"
    ra = rec.get("relation_accuracy")
    if isinstance(ra, dict) and ra.get("n_relations", 0) > 0:
        return ra.get("accuracy", float("nan")), "rel"
    return rec.get("clip_score", float("nan")), "CLIP"


def _caption_for(rec: Optional[dict]) -> str:
    if not rec:
        return "(no score)"
    parts = []
    if "clip_score" in rec:
        parts.append(f"CLIP {rec['clip_score']:.3f}")
    ab = rec.get("attr_binding")
    if isinstance(ab, dict) and ab.get("n_pairs", 0) > 0:
        acc = ab.get("accuracy", float("nan"))
        mark = "\u2713" if acc >= 0.999 else ("\u2717" if acc <= 0.001 else f"{acc:.2f}")
        parts.append(f"attr {mark}")
    ra = rec.get("relation_accuracy")
    if isinstance(ra, dict) and ra.get("n_relations", 0) > 0:
        acc = ra.get("accuracy", float("nan"))
        mark = "\u2713" if acc >= 0.999 else ("\u2717" if acc <= 0.001 else f"{acc:.2f}")
        parts.append(f"rel {mark}")
    return "  ".join(parts) if parts else "(no score)"


def _find_image(img_dir: Path, idx: int) -> Optional[Path]:
    matches = sorted(img_dir.glob(f"prompt_{idx:03d}_*.png"))
    if matches:
        return matches[0]
    allpng = sorted(img_dir.glob("*.png"))
    return allpng[idx] if idx < len(allpng) else None


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _select_examples(single_tr: dict, poe_tr: dict, args) -> list[int]:
    if args.indices:
        return args.indices

    common = sorted(set(single_tr) & set(poe_tr))
    if not common:
        # No overlap (e.g. no scores) — fall back to whatever poe has.
        return sorted(poe_tr)[:args.n]

    if args.select == "first":
        return common[:args.n]
    if args.select == "random":
        import random
        random.seed(args.seed)
        return sorted(random.sample(common, min(args.n, len(common))))

    # best: largest (PoE - LLaDA) margin on the decisive metric.
    def _delta(i):
        s, _ = _metric(single_tr.get(i))
        p, _ = _metric(poe_tr.get(i))
        if s != s or p != p:      # NaN guard
            return float("-inf")
        return p - s
    ranked = sorted(common, key=_delta, reverse=True)
    return sorted(ranked[:args.n], key=lambda i: -_delta(i))  # keep best-first


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build(args, outputs: Path, out_stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rq5 = outputs / args.backbone / "rq5"
    single_dir, poe_dir = rq5 / "llada_single", rq5 / "llada_poe"
    single_tr = _load_trace(single_dir / "trace.jsonl")
    poe_tr = _load_trace(poe_dir / "trace.jsonl")

    if not (single_dir.exists() and poe_dir.exists()):
        logger.error("RQ5 llada_single / llada_poe dirs missing under %s — run RQ5 first.", rq5)
        return

    indices = _select_examples(single_tr, poe_tr, args)
    if not indices:
        logger.error("No examples selected — check RQ5 outputs.")
        return

    cols = args.cols
    if args.columns:
        # Side-by-side variant: rows = prompts, 2 columns (LLaDA | PoE).
        _build_columns(args, indices, single_dir, poe_dir, single_tr, poe_tr,
                       outputs, out_stem, plt, Image)
        return

    # Paired-rows layout: blocks of [2 rows x cols]. Top=LLaDA, bottom=PoE.
    n = len(indices)
    n_blocks = math.ceil(n / cols)
    fig_rows = n_blocks * 2
    fig, axes = plt.subplots(
        fig_rows, cols,
        figsize=(3.1 * cols, 3.5 * fig_rows),
        squeeze=False,
    )

    for b in range(n_blocks):
        top_r, bot_r = b * 2, b * 2 + 1
        for c in range(cols):
            ei = b * cols + c
            ax_top, ax_bot = axes[top_r][c], axes[bot_r][c]
            for ax in (ax_top, ax_bot):
                ax.axis("off")
            if ei >= n:
                continue
            idx = indices[ei]

            # Top = LLaDA
            _place(ax_top, _find_image(single_dir, idx), single_tr.get(idx),
                   Image, _caption_for)
            # Bottom = LLaDA + PoE
            _place(ax_bot, _find_image(poe_dir, idx), poe_tr.get(idx),
                   Image, _caption_for)

            # Prompt as a small title over the top image of each column.
            prompt = (poe_tr.get(idx) or single_tr.get(idx) or {}).get("prompt", f"prompt {idx}")
            prompt = prompt if len(prompt) <= 34 else prompt[:31] + "..."
            ax_top.set_title(prompt, fontsize=8.5, fontstyle="italic", pad=4)

        # Row-group labels on the far left.
        axes[top_r][0].annotate(_TOP_LABEL, xy=(0, 0.5), xytext=(-14, 0),
                                textcoords="offset points", xycoords="axes fraction",
                                ha="right", va="center", fontsize=11,
                                fontweight="bold", rotation=90)
        axes[bot_r][0].annotate(_BOTTOM_LABEL, xy=(0, 0.5), xytext=(-14, 0),
                                textcoords="offset points", xycoords="axes fraction",
                                ha="right", va="center", fontsize=11,
                                fontweight="bold", rotation=90)

    caption = _make_caption(args, indices, single_tr, poe_tr)
    fig.suptitle("LLaDA vs. LLaDA + Product-of-Experts composition",
                 fontsize=13, fontweight="bold", y=0.995)
    # Render caption as a text box under the figure.
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=8.5,
             wrap=True, fontstyle="normal")
    fig.tight_layout(rect=[0.03, 0.04, 1, 0.98])

    _save(fig, out_stem, caption, plt)


def _build_columns(args, indices, single_dir, poe_dir, single_tr, poe_tr,
                   outputs, out_stem, plt, Image):
    """Side-by-side variant: rows = prompts, col0 = LLaDA, col1 = PoE."""
    n = len(indices)
    fig, axes = plt.subplots(n, 2, figsize=(6.6, 3.4 * n), squeeze=False)
    axes[0][0].set_title(_TOP_LABEL, fontsize=12, fontweight="bold", pad=8)
    axes[0][1].set_title(_BOTTOM_LABEL, fontsize=12, fontweight="bold", pad=8)
    for r, idx in enumerate(indices):
        _place(axes[r][0], _find_image(single_dir, idx), single_tr.get(idx),
               Image, _caption_for)
        _place(axes[r][1], _find_image(poe_dir, idx), poe_tr.get(idx),
               Image, _caption_for)
        for c in (0, 1):
            axes[r][c].axis("off")
        prompt = (poe_tr.get(idx) or single_tr.get(idx) or {}).get("prompt", f"prompt {idx}")
        wrapped = prompt if len(prompt) <= 40 else prompt[:37] + "..."
        axes[r][0].annotate(wrapped, xy=(0, 0.5), xytext=(-10, 0),
                            textcoords="offset points", xycoords="axes fraction",
                            ha="right", va="center", fontsize=8.5, fontstyle="italic")
    caption = _make_caption(args, indices, single_tr, poe_tr)
    fig.suptitle("LLaDA vs. LLaDA + PoE", fontsize=13, fontweight="bold", y=0.997)
    fig.text(0.5, 0.004, caption, ha="center", va="bottom", fontsize=8.5, wrap=True)
    fig.tight_layout(rect=[0.05, 0.03, 1, 0.98])
    _save(fig, out_stem, caption, plt)


def _place(ax, img_path, rec, Image, cap_fn):
    ax.axis("off")
    if img_path and img_path.exists():
        try:
            ax.imshow(Image.open(img_path).convert("RGB"))
        except Exception:
            ax.text(0.5, 0.5, "load error", ha="center", va="center",
                    fontsize=8, transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "(missing)", ha="center", va="center",
                fontsize=9, color="#999", transform=ax.transAxes)
    ax.set_xlabel(cap_fn(rec), fontsize=8.5)


def _make_caption(args, indices, single_tr, poe_tr) -> str:
    """Paper-style caption body, also written to a .txt for LaTeX."""
    sel = {"best": "selected as the cases where PoE improves most over single-pass "
                   "LLaDA on the compositional metric",
           "first": "the first prompts of the compositional set",
           "random": f"a random sample (seed {args.seed})"}.get(
               "manual" if args.indices else args.select,
               "selected examples")
    # Average deltas for the caption's quantitative hook.
    deltas = []
    for i in indices:
        s, _ = _metric(single_tr.get(i))
        p, _ = _metric(poe_tr.get(i))
        if s == s and p == p:
            deltas.append(p - s)
    hook = ""
    if deltas:
        mean_d = sum(deltas) / len(deltas)
        hook = (f" Across the {len(indices)} examples shown, PoE improves the "
                f"compositional metric by {mean_d:+.3f} on average.")
    return (
        f"Figure: Qualitative comparison of single-pass LLaDA refinement (top) "
        f"against LLaDA with language-side Product-of-Experts composition (bottom) "
        f"on the {args.backbone.upper()} backbone. Each column is one compositional "
        f"prompt (shown above); the same prompt appears top and bottom. Examples are "
        f"{sel}. Per-image annotations give CLIPScore and attribute/relation "
        f"binding accuracy (\u2713 = bound, \u2717 = failed).{hook} PoE composes the "
        f"decomposed constraints jointly \u2014 an operation the autoregressive baseline "
        f"cannot perform."
    )


def _save(fig, out_stem: Path, caption: str, plt) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), bbox_inches="tight", dpi=200)
    (out_stem.parent / f"{out_stem.name}_caption.txt").write_text(caption, encoding="utf-8")
    plt.close(fig)
    logger.info("Wrote %s.png / .pdf (+ caption .txt)", out_stem)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="LLaDA vs LLaDA+PoE paired figure")
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--backbone", default="sdxl")
    p.add_argument("--n", type=int, default=6, help="Number of example prompts (~6-9)")
    p.add_argument("--cols", type=int, default=3, help="Columns per block (paired-rows layout)")
    p.add_argument("--select", default="best", choices=["best", "first", "random"])
    p.add_argument("--indices", type=int, nargs="*", help="Explicit prompt indices")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--columns", action="store_true",
                   help="Side-by-side variant (rows=prompts, cols=LLaDA|PoE)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    outputs = Path(args.outputs)
    out_stem = Path(args.out) if args.out else outputs / "plots" / "figures" / "poe_grid"
    build(args, outputs, out_stem)


if __name__ == "__main__":
    main()

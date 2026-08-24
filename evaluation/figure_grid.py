"""
evaluation/figure_grid.py

Dissertation-ready qualitative comparison figures.

Lays out generated images as a grid: one PROMPT per row, one CONDITION per
column, with each image annotated by its metric score. Reads images + trace
scores already on disk (no regeneration).

Four comparison presets (columns):
    pipeline  : raw / ar / llada                     (RQ2 refinement story)
    poe       : raw / ar / llada_single / llada_poe  (RQ5 capstone)
    backbone  : sd21 / sdxl (same pipeline)          (encoder-capacity)
    custom    : any list of (label, image_dir, trace) you pass

Prompt selection:
    --select first|best|worst|random  (auto)
    --indices 0 3 7                    (manual override)
    best/worst rank by the chosen metric on the LAST column (the method whose
    wins/losses you most want to show), so "best" = where that method scores
    highest, "worst" = its failures.

Annotations:
    Each image is captioned with CLIPScore and a binding verdict (attr / rel
    accuracy, shown as a fraction or check/cross), read from the trace.

Usage:
    # RQ2 pipeline comparison, 6 best LLaDA cases from color_binding on sd21
    python -m evaluation.figure_grid --preset pipeline --backbone sd21 \\
        --set color_binding --select best --n 6

    # RQ5 PoE capstone, worst 5 (PoE failures — honest reporting)
    python -m evaluation.figure_grid --preset poe --backbone sd21 \\
        --select worst --n 5

    # backbone comparison for the llada pipeline
    python -m evaluation.figure_grid --preset backbone --pipeline llada_clip \\
        --set color_binding --select random --n 6

    # explicit prompts
    python -m evaluation.figure_grid --preset pipeline --backbone sd21 \\
        --set color_binding --indices 0 5 12 40
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trace loading + score extraction
# ---------------------------------------------------------------------------

def _load_trace(trace_path: Path) -> dict[int, dict]:
    """Load a trace.jsonl into {idx: record}. Empty dict if missing."""
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


def _score_caption(rec: Optional[dict]) -> str:
    """One-line score caption for an image from its trace record."""
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


def _rank_metric(rec: Optional[dict]) -> float:
    """Scalar used to rank prompts for best/worst selection."""
    if not rec:
        return float("-inf")
    # Prefer attribute binding, then relation, then CLIPScore.
    ab = rec.get("attr_binding")
    if isinstance(ab, dict) and ab.get("n_pairs", 0) > 0:
        return ab.get("accuracy", 0.0)
    ra = rec.get("relation_accuracy")
    if isinstance(ra, dict) and ra.get("n_relations", 0) > 0:
        return ra.get("accuracy", 0.0)
    return rec.get("clip_score", 0.0)


# ---------------------------------------------------------------------------
# Column specification per preset
# ---------------------------------------------------------------------------

def _columns_for_preset(args, outputs: Path) -> list[tuple[str, Path, Path]]:
    """
    Return [(column_label, image_dir, trace_path), ...] for the chosen preset.

    Image dir layout (from the RQ runners):
        RQ2 : outputs/<backbone>/rq2/<pipeline>/<set>/  (images + trace.jsonl)
        RQ5 : outputs/<backbone>/rq5/<condition>/       (images + trace.jsonl)
    """
    preset = args.preset

    if preset == "pipeline":
        bb = args.backbone
        s = args.set
        base = outputs / bb / "rq2"
        cols = []
        for pipe, label in [("raw_clip", "raw"), ("ar_clip", "AR"),
                            ("llada_clip", "LLaDA")]:
            d = base / pipe / s
            cols.append((label, d, d / "trace.jsonl"))
        return cols

    if preset == "poe":
        bb = args.backbone
        base = outputs / bb / "rq5"
        cols = []
        for cond, label in [("raw", "raw"), ("ar", "AR"),
                            ("llada_single", "LLaDA"), ("llada_poe", "LLaDA+PoE")]:
            d = base / cond
            cols.append((label, d, d / "trace.jsonl"))
        return cols

    if preset == "backbone":
        pipe = args.pipeline
        s = args.set
        cols = []
        for bb in ["sd21", "sdxl"]:
            d = outputs / bb / "rq2" / pipe / s
            cols.append((bb, d, d / "trace.jsonl"))
        return cols

    raise ValueError(f"Unknown preset {preset!r}")


def _find_image(img_dir: Path, idx: int) -> Optional[Path]:
    """Find the image for a prompt index, tolerant of cfg/seed in the name."""
    # Filenames look like prompt_000_cfg7.5_seed42.png
    matches = sorted(img_dir.glob(f"prompt_{idx:03d}_*.png"))
    if matches:
        return matches[0]
    # Fallback: any png at this ordinal position.
    allpng = sorted(img_dir.glob("*.png"))
    return allpng[idx] if idx < len(allpng) else None


# ---------------------------------------------------------------------------
# Prompt selection
# ---------------------------------------------------------------------------

def _select_indices(args, columns) -> list[int]:
    """Pick which prompt indices to show, per --indices or --select."""
    if args.indices:
        return args.indices

    # Rank by the LAST column's metric (the method whose wins/losses to show).
    _, _, last_trace = columns[-1]
    trace = _load_trace(last_trace)
    if not trace:
        # No scores — fall back to first-N over whatever images exist.
        _, last_dir, _ = columns[-1]
        n_imgs = len(list(last_dir.glob("*.png")))
        return list(range(min(args.n, n_imgs)))

    idxs = sorted(trace.keys())
    if args.select == "first":
        return idxs[:args.n]
    if args.select == "random":
        random.seed(args.seed)
        return sorted(random.sample(idxs, min(args.n, len(idxs))))
    # best / worst by rank metric
    ranked = sorted(idxs, key=lambda i: _rank_metric(trace.get(i)),
                    reverse=(args.select == "best"))
    return sorted(ranked[:args.n])


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def build_figure(args, outputs: Path, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    columns = _columns_for_preset(args, outputs)
    indices = _select_indices(args, columns)
    if not indices:
        logger.error("No prompt indices selected — check that images exist.")
        return

    # Preload traces per column for captions + row prompt text.
    col_traces = [_load_trace(t) for (_, _, t) in columns]

    n_rows, n_cols = len(indices), len(columns)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 3.3 * n_rows),
        squeeze=False,
    )

    # Column headers.
    for c, (label, _, _) in enumerate(columns):
        axes[0][c].set_title(label, fontsize=13, fontweight="bold", pad=8)

    for r, idx in enumerate(indices):
        # Row label = the prompt (from any column's trace that has it).
        prompt_text = None
        for tr in col_traces:
            if idx in tr and tr[idx].get("prompt"):
                prompt_text = tr[idx]["prompt"]
                break
        if prompt_text is None:
            prompt_text = f"prompt {idx}"

        for c, (label, img_dir, _) in enumerate(columns):
            ax = axes[r][c]
            ax.axis("off")
            img_path = _find_image(img_dir, idx)
            if img_path and img_path.exists():
                try:
                    ax.imshow(Image.open(img_path).convert("RGB"))
                except Exception as exc:
                    ax.text(0.5, 0.5, "load error", ha="center", va="center",
                            fontsize=8, transform=ax.transAxes)
                    logger.debug("Failed to load %s (%s)", img_path, exc)
            else:
                ax.text(0.5, 0.5, "(missing)", ha="center", va="center",
                        fontsize=9, color="#999", transform=ax.transAxes)

            # Per-image score caption.
            cap = _score_caption(col_traces[c].get(idx))
            ax.set_xlabel(cap, fontsize=9)

        # Row prompt as a left-side annotation on the first axis.
        wrapped = prompt_text if len(prompt_text) <= 40 else prompt_text[:37] + "..."
        axes[r][0].annotate(
            wrapped, xy=(0, 0.5), xytext=(-10, 0),
            textcoords="offset points", xycoords="axes fraction",
            ha="right", va="center", fontsize=9, fontstyle="italic",
            rotation=0, wrap=True,
        )

    title = args.title or _default_title(args)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0.04, 0, 1, 0.98])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_path.with_suffix(f".{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info("Wrote %s.png / .pdf (%d rows x %d cols)",
                out_path, n_rows, n_cols)


def _default_title(args) -> str:
    sel = "manual" if args.indices else args.select
    if args.preset == "pipeline":
        return f"Pipeline comparison - {args.set} ({args.backbone}, {sel})"
    if args.preset == "poe":
        return f"PoE composition - RQ5 ({args.backbone}, {sel})"
    if args.preset == "backbone":
        return f"SD 2.1 vs SDXL - {args.pipeline} / {args.set} ({sel})"
    return "Comparison"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Dissertation image comparison grids")
    p.add_argument("--preset", required=True,
                   choices=["pipeline", "poe", "backbone"],
                   help="Which comparison the columns make")
    p.add_argument("--outputs", default="outputs", help="Outputs root")
    p.add_argument("--backbone", default="sd21", help="Backbone (pipeline/poe presets)")
    p.add_argument("--pipeline", default="llada_clip", help="Pipeline (backbone preset)")
    p.add_argument("--set", default="color_binding", help="Prompt set (pipeline/backbone)")
    p.add_argument("--select", default="first",
                   choices=["first", "best", "worst", "random"],
                   help="Auto prompt selection strategy")
    p.add_argument("--n", type=int, default=6, help="How many prompts (rows)")
    p.add_argument("--indices", type=int, nargs="*",
                   help="Explicit prompt indices (overrides --select)")
    p.add_argument("--seed", type=int, default=0, help="Seed for --select random")
    p.add_argument("--title", default=None, help="Override figure title")
    p.add_argument("--out", default=None, help="Output path stem (no extension)")
    args = p.parse_args()

    outputs = Path(args.outputs)
    if args.out:
        out_path = Path(args.out)
    else:
        sel = "manual" if args.indices else args.select
        stem = f"grid_{args.preset}_{sel}"
        out_path = outputs / "plots" / "figures" / stem

    build_figure(args, outputs, out_path)


if __name__ == "__main__":
    main()

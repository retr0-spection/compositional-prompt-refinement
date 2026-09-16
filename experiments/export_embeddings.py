"""
experiments/export_embeddings.py

Re-encode the RQ1 prompt sets and save the pooled per-prompt conditioning
embeddings to disk, so the embedding-separation finding (RQ1) can be
illustrated with a 2D projection.

RQ1 computes the separation-gain scalar in the full embedding space but does
not persist the embedding vectors themselves. This script recovers them by
re-encoding each prompt through the same pipelines used in RQ1 (raw / AR /
LLaDA), applying the same EOS pooling as
evaluation.embedding_analysis.compute_embedding_separation, and writes one
matrix per pipeline.

The saved vectors feed evaluation.plotting.plot_embedding_pca, which projects
them to two dimensions with PCA (a linear, deterministic projection that,
unlike t-SNE or UMAP, cannot manufacture apparent clusters). The projection is
an ILLUSTRATION of the separation-gain metric, which remains the rigorous
quantity, computed in the full space.

Output (per backbone):
    outputs/<backbone>/rq1/embeddings/<pipeline>.npz
        arrays: 'pooled' (n_prompts, hidden), 'prompts' (n_prompts,)

Usage (on a GPU node; needs the backbone text encoder, and Ollama for the AR
and LLaDA rewrites unless their caches are already populated):
    python experiments/export_embeddings.py \\
        --config t2i.backbone=sdxl t2i.model_id=... t2i.resolution=1024 \\
                 t2i.prediction_type=epsilon
"""

from __future__ import annotations

import argparse
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


def _pool_eos(embeddings, attention_mask=None):
    """
    EOS-pool a (n, seq_len, hidden) tensor to (n, hidden), matching
    evaluation.embedding_analysis.compute_embedding_separation.
    """
    import torch
    if attention_mask is not None:
        lengths = attention_mask.sum(dim=1).long()
        eos_idx = (lengths - 1).clamp(min=0)
        pooled = embeddings[torch.arange(embeddings.size(0)), eos_idx]
    else:
        pooled = embeddings[:, -1, :]
    return pooled


def export_for_pipeline(pipeline, prompts, out_dir):
    """
    Encode `prompts` through one pipeline, EOS-pool, and save to
    out_dir/<pipeline.name>.npz. Returns the pooled matrix.
    """
    import numpy as np

    logger.info("[export] encoding %d prompts through %s", len(prompts), pipeline.name)
    enc_results = pipeline.encode_batch(prompts)

    pooled_rows = []
    for enc in enc_results:
        emb = enc.embedding  # (1, seq_len, hidden) or (seq_len, hidden)
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)
        mask = getattr(enc, "attention_mask", None)
        p = _pool_eos(emb, mask)  # (1, hidden)
        pooled_rows.append(p.squeeze(0).float().cpu().numpy())

    pooled = np.stack(pooled_rows, axis=0)  # (n, hidden)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{pipeline.name}.npz",
             pooled=pooled, prompts=np.array(prompts, dtype=object))
    logger.info("[export] wrote %s (%s)", out_dir / f"{pipeline.name}.npz", pooled.shape)
    return pooled


def main():
    parser = argparse.ArgumentParser(description="Export RQ1 conditioning embeddings")
    parser.add_argument("--set", default=None,
                        help="Prompt set to encode (default: union of RQ1 sets)")
    parser.add_argument("--config", nargs="*", metavar="KEY=VALUE")
    args = parser.parse_args()

    from experiments.run_experiment import load_config, build_pipelines
    from utils.prompt_io import load_prompts

    cfg = load_config(args.config)
    backbone = cfg.get("t2i", {}).get("backbone", "sd21")

    # Collect prompts: the RQ1 sets (or a named set).
    if args.set:
        prompts = load_prompts(args.set)
    else:
        set_names = cfg.get("eval_prompt_sets", {}).get(
            "rq1", ["color_binding", "shape_binding", "texture_binding",
                    "spatial_relations"])
        prompts = []
        for s in set_names:
            prompts.extend(load_prompts(s))
    logger.info("Exporting embeddings for backbone=%s over %d prompts",
                backbone, len(prompts))

    pipelines = build_pipelines(cfg, dry_run=False)
    out_dir = Path(cfg.get("output_dir", "outputs")) / backbone / "rq1" / "embeddings"

    for pipeline in pipelines:
        try:
            export_for_pipeline(pipeline, prompts, out_dir)
        except Exception as exc:
            logger.warning("[export] %s failed (%s)", pipeline.name, exc)

    logger.info("Done. Embeddings in %s", out_dir)


if __name__ == "__main__":
    main()

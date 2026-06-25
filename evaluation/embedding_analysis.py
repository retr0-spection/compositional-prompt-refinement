"""
evaluation/embedding_analysis.py

RQ1: Does diffusion-based refinement produce more semantically structured
conditioning signals than raw prompts?

Two analyses:
1. Entity/relation density — counts how many entity/attribute/relation
   tokens survive in the rewritten prompt vs the raw prompt.
2. CLIP embedding cosine separation — measures whether different prompts
   produce more separated embeddings after rewriting (richer = more distinct).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Simple entity/relation keyword lists (extend or replace with spaCy NER)
_ATTRIBUTE_TOKENS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "white", "black", "grey", "gray", "brown", "gold", "silver",
    "large", "small", "big", "tiny", "tall", "short", "long", "wide",
    "wooden", "metal", "glass", "ceramic", "fluffy", "smooth", "rough",
    "bright", "dark", "shiny", "matte", "transparent", "opaque",
]

_RELATION_TOKENS = [
    "left", "right", "above", "below", "behind", "front", "top", "under",
    "beside", "next", "near", "between", "inside", "outside", "on", "off",
    "against", "around", "through", "across", "along",
]


def count_semantic_tokens(text: str) -> dict[str, int]:
    """
    Count attribute and relation tokens in a text string.

    Returns
    -------
    dict with keys: n_attribute_tokens, n_relation_tokens, total_words
    """
    words = text.lower().split()
    n_attr = sum(1 for w in words if w in _ATTRIBUTE_TOKENS)
    n_rel = sum(1 for w in words if w in _RELATION_TOKENS)
    return {
        "n_attribute_tokens": n_attr,
        "n_relation_tokens": n_rel,
        "total_words": len(words),
        "semantic_density": (n_attr + n_rel) / len(words) if words else 0.0,
    }


def analyse_semantic_density(
    raw_prompts: list[str],
    rewritten_prompts: list[str],
    pipeline_name: str,
) -> dict[str, float]:
    """
    Compare semantic density between raw and rewritten prompts.

    Returns a flat dict of aggregate stats for W&B logging.
    """
    raw_stats = [count_semantic_tokens(p) for p in raw_prompts]
    rw_stats = [count_semantic_tokens(p) for p in rewritten_prompts]
    n = len(raw_prompts)

    def avg(lst, key):
        return sum(d[key] for d in lst) / n if n else 0.0

    return {
        f"{pipeline_name}/raw_attr_density":     avg(raw_stats, "n_attribute_tokens"),
        f"{pipeline_name}/raw_rel_density":      avg(raw_stats, "n_relation_tokens"),
        f"{pipeline_name}/raw_semantic_density": avg(raw_stats, "semantic_density"),
        f"{pipeline_name}/rw_attr_density":      avg(rw_stats, "n_attribute_tokens"),
        f"{pipeline_name}/rw_rel_density":       avg(rw_stats, "n_relation_tokens"),
        f"{pipeline_name}/rw_semantic_density":  avg(rw_stats, "semantic_density"),
        f"{pipeline_name}/attr_density_gain":    avg(rw_stats, "n_attribute_tokens") - avg(raw_stats, "n_attribute_tokens"),
        f"{pipeline_name}/rel_density_gain":     avg(rw_stats, "n_relation_tokens") - avg(raw_stats, "n_relation_tokens"),
    }


def compute_embedding_separation(
    embeddings: torch.Tensor,
) -> dict[str, float]:
    """
    Compute pairwise cosine similarity statistics for a batch of embeddings.

    Uses the [EOS] / pooled representation (mean over seq dimension) for
    comparison, since per-token embeddings are not directly comparable.

    Parameters
    ----------
    embeddings : torch.Tensor
        Shape: (n_prompts, seq_len, hidden_dim)

    Returns
    -------
    dict with mean_pairwise_similarity, std_pairwise_similarity,
    mean_pairwise_distance (1 - similarity).
    """
    # Pool over sequence dimension → (n, hidden_dim)
    pooled = embeddings.mean(dim=1)
    pooled = F.normalize(pooled, dim=-1)

    # Pairwise cosine similarity matrix
    sim_matrix = torch.mm(pooled, pooled.T)           # (n, n)
    n = pooled.shape[0]

    # Exclude diagonal (self-similarity = 1.0)
    mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
    off_diag = sim_matrix[mask]

    mean_sim = float(off_diag.mean().item())
    std_sim = float(off_diag.std().item())

    return {
        "mean_pairwise_cosine_similarity": mean_sim,
        "std_pairwise_cosine_similarity": std_sim,
        "mean_pairwise_distance": 1.0 - mean_sim,
    }


def compare_embedding_separation(
    raw_embeddings: torch.Tensor,
    rewritten_embeddings: torch.Tensor,
    pipeline_name: str,
) -> dict[str, float]:
    """
    Compare pairwise embedding separation before and after rewriting.

    Higher pairwise distance after rewriting means the encoder can
    better distinguish different prompts — a proxy for richer conditioning.
    """
    raw_stats = compute_embedding_separation(raw_embeddings)
    rw_stats = compute_embedding_separation(rewritten_embeddings)

    out = {}
    for k, v in raw_stats.items():
        out[f"{pipeline_name}/raw_{k}"] = v
    for k, v in rw_stats.items():
        out[f"{pipeline_name}/rw_{k}"] = v

    # Separation gain: positive means rewritten prompts are more distinct
    out[f"{pipeline_name}/separation_gain"] = (
        rw_stats["mean_pairwise_distance"] - raw_stats["mean_pairwise_distance"]
    )
    return out

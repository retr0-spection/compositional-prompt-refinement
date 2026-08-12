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

import json
import logging
import re
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Fallback keyword lists — used only when the LLM extractor is unavailable.
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

# ---------------------------------------------------------------------------
# LLM-based semantic extraction (Ollama)
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = (
    "Count the visual attributes and spatial relations in this image caption.\n"
    "- Attributes = words describing an object's colour, size, shape, material, "
    "or texture (e.g. red, tiny, wooden, striped).\n"
    "- Relations = words/phrases describing spatial arrangement between objects "
    "(e.g. left of, above, beside, on top of, behind).\n"
    "Count every occurrence, including synonyms and multi-word phrases.\n"
    'Respond ONLY with JSON: {{"attributes": <int>, "relations": <int>}}\n\n'
    "Caption: {text}"
)

_JSON_RE = re.compile(r'\{[^{}]*"attributes"[^{}]*\}', re.DOTALL)


class SemanticCounter:
    """
    Counts attributes and relations in text, preferring an Ollama LLM and
    falling back to keyword matching when the LLM is unavailable.

    The LLM catches attributes/relations the fixed keyword lists miss
    (crimson, enormous, velvet, 'perched atop', ...), giving a truer count
    of how much compositional content a rewrite actually added.

    Results are cached per (text) so repeated prompts across RQ1 pipelines
    don't re-query the LLM.
    """

    def __init__(
        self,
        use_llm: bool = True,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ) -> None:
        self.use_llm = use_llm
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._cache: dict[str, dict[str, int]] = {}
        self._llm_ok: Optional[bool] = None  # None = untested

    # -- keyword fallback ------------------------------------------------

    @staticmethod
    def _keyword_count(text: str) -> dict[str, int]:
        words = text.lower().split()
        return {
            "attributes": sum(1 for w in words if w in _ATTRIBUTE_TOKENS),
            "relations": sum(1 for w in words if w in _RELATION_TOKENS),
        }

    # -- LLM path --------------------------------------------------------

    def _llm_count(self, text: str) -> Optional[dict[str, int]]:
        """Query Ollama; return None on any failure so caller can fall back."""
        import requests
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": _EXTRACT_PROMPT.format(text=text),
                    "stream": False,
                    "options": {"temperature": 0.0},
                    "format": "json",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            body = resp.json().get("response", "")
            # With format=json Ollama returns clean JSON, but guard anyway.
            match = _JSON_RE.search(body) or re.search(r"\{.*\}", body, re.DOTALL)
            data = json.loads(match.group(0) if match else body)
            return {
                "attributes": int(data.get("attributes", 0)),
                "relations": int(data.get("relations", 0)),
            }
        except Exception as exc:
            logger.debug("LLM count failed for %r (%s)", text[:50], exc)
            return None

    # -- public ----------------------------------------------------------

    def count(self, text: str) -> dict[str, int]:
        """
        Return {attributes, relations, total_words} for a single text.

        Tries the LLM once; if the first call fails, disables the LLM for the
        rest of this run and uses keyword counting throughout (avoids 500
        timeouts per prompt when Ollama isn't up).
        """
        if text in self._cache:
            counts = self._cache[text]
        else:
            counts = None
            if self.use_llm and self._llm_ok is not False:
                counts = self._llm_count(text)
                if counts is None and self._llm_ok is None:
                    logger.warning(
                        "Ollama unavailable for semantic counting — "
                        "falling back to keyword lists for this run."
                    )
                    self._llm_ok = False
                elif counts is not None:
                    self._llm_ok = True
            if counts is None:
                counts = self._keyword_count(text)
            self._cache[text] = counts

        return {
            "n_attribute_tokens": counts["attributes"],
            "n_relation_tokens": counts["relations"],
            "total_words": len(text.split()),
            "semantic_density": (
                (counts["attributes"] + counts["relations"]) / len(text.split())
                if text.split() else 0.0
            ),
        }


# Module-level default counter (LLM on). Override in analyse_semantic_density
# for tests or when Ollama is known to be down.
_DEFAULT_COUNTER = SemanticCounter(use_llm=True)


def count_semantic_tokens(text: str, counter: Optional[SemanticCounter] = None) -> dict:
    """
    Count attribute and relation tokens in a text string.

    Uses the LLM-backed SemanticCounter by default (keyword fallback if the
    LLM is unavailable). Pass a custom counter to force keyword-only:
        count_semantic_tokens(text, SemanticCounter(use_llm=False))

    Returns keys: n_attribute_tokens, n_relation_tokens, total_words,
    semantic_density.
    """
    return (counter or _DEFAULT_COUNTER).count(text)


def analyse_semantic_density(
    raw_prompts: list[str],
    rewritten_prompts: list[str],
    pipeline_name: str,
    counter: Optional["SemanticCounter"] = None,
) -> dict[str, float]:
    """
    Compare semantic content between raw and rewritten prompts.

    Naming: attribute/relation values are COUNTS (per-prompt averages of raw
    occurrence counts). semantic_density is the only true density — the
    fraction (attributes + relations) / total_words.

    Returns a flat dict of aggregate stats for W&B logging.
    """
    counter = counter or _DEFAULT_COUNTER
    raw_stats = [counter.count(p) for p in raw_prompts]
    rw_stats = [counter.count(p) for p in rewritten_prompts]
    n = len(raw_prompts)

    def avg(lst, key):
        return sum(d[key] for d in lst) / n if n else 0.0

    return {
        # counts (renamed from *_density — these are counts, not ratios)
        f"{pipeline_name}/raw_attr_count":       avg(raw_stats, "n_attribute_tokens"),
        f"{pipeline_name}/raw_rel_count":        avg(raw_stats, "n_relation_tokens"),
        f"{pipeline_name}/rw_attr_count":        avg(rw_stats, "n_attribute_tokens"),
        f"{pipeline_name}/rw_rel_count":         avg(rw_stats, "n_relation_tokens"),
        f"{pipeline_name}/attr_count_gain":      avg(rw_stats, "n_attribute_tokens") - avg(raw_stats, "n_attribute_tokens"),
        f"{pipeline_name}/rel_count_gain":       avg(rw_stats, "n_relation_tokens") - avg(raw_stats, "n_relation_tokens"),
        # true density (the ratio)
        f"{pipeline_name}/raw_semantic_density": avg(raw_stats, "semantic_density"),
        f"{pipeline_name}/rw_semantic_density":  avg(rw_stats, "semantic_density"),
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

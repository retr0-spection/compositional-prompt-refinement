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
# The LLM extracts a lightweight SCENE GRAPH — objects, each object's
# attributes, and inter-object relations — rather than just counting tokens.
# This gives binding-level structure (red→cat, cat left-of dog), from which
# the old counts are derived as a view. Keyword matching remains as a fallback
# when Ollama is unavailable.

_EXTRACT_PROMPT = (
    "Extract the visual scene structure from this image caption as JSON.\n"
    "- objects: list of distinct physical objects (nouns).\n"
    "- attributes: list of [object, attribute] pairs, where attribute is a "
    "colour, size, shape, material, or texture bound to that object.\n"
    "- relations: list of [object1, relation, object2] triples describing "
    "spatial arrangement (left of, above, beside, on top of, behind, etc).\n"
    "Include synonyms and multi-word phrases. Use the exact object words from "
    "the caption.\n"
    'Respond ONLY with JSON of the form: '
    '{{"objects": [...], "attributes": [[obj, attr], ...], '
    '"relations": [[obj1, rel, obj2], ...]}}\n\n'
    "Caption: {text}"
)

_JSON_RE = re.compile(r'\{.*"objects".*\}', re.DOTALL)


class SemanticExtractor:
    """
    Extracts a scene graph (objects / attribute-bindings / relations) from
    text using an Ollama LLM, with a keyword-count fallback.

    Beyond raw counts, this captures BINDING structure — which attribute is
    tied to which object, and how objects relate spatially — which is the
    compositional signal RQ1 actually cares about. Counts are derived as a
    view (len of each list) for back-compatibility.

    Each result records `method` ('llm' | 'keyword') so a run can report what
    fraction fell back, turning silent degradation into a reported number.

    Results are cached per text so repeated prompts across pipelines don't
    re-query the LLM.
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
        self._cache: dict[str, dict] = {}
        self._llm_ok: Optional[bool] = None  # None = untested

    # -- keyword fallback ------------------------------------------------

    @staticmethod
    def _keyword_extract(text: str) -> dict:
        """Fallback: keyword counts only, no true binding structure."""
        words = text.lower().split()
        attrs = [w for w in words if w in _ATTRIBUTE_TOKENS]
        rels = [w for w in words if w in _RELATION_TOKENS]
        return {
            # No object/binding structure available from keywords — approximate
            # attributes as unbound [?, attr] and relations as [?, rel, ?].
            "objects": [],
            "attributes": [["?", a] for a in attrs],
            "relations": [["?", r, "?"] for r in rels],
            "method": "keyword",
        }

    # -- LLM path --------------------------------------------------------

    def _llm_extract(self, text: str) -> Optional[dict]:
        """Query Ollama for a scene graph; None on any failure."""
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
            match = _JSON_RE.search(body) or re.search(r"\{.*\}", body, re.DOTALL)
            data = json.loads(match.group(0) if match else body)

            # Validate + coerce structure; reject malformed -> fall back.
            objs = data.get("objects", [])
            attrs = data.get("attributes", [])
            rels = data.get("relations", [])
            if not isinstance(objs, list) or not isinstance(attrs, list) \
               or not isinstance(rels, list):
                return None
            # Keep only well-formed entries.
            attrs = [a for a in attrs if isinstance(a, (list, tuple)) and len(a) == 2]
            rels = [r for r in rels if isinstance(r, (list, tuple)) and len(r) == 3]
            return {
                "objects": [str(o) for o in objs],
                "attributes": [[str(a[0]), str(a[1])] for a in attrs],
                "relations": [[str(r[0]), str(r[1]), str(r[2])] for r in rels],
                "method": "llm",
            }
        except Exception as exc:
            logger.debug("LLM extract failed for %r (%s)", text[:50], exc)
            return None

    # -- public ----------------------------------------------------------

    def extract(self, text: str) -> dict:
        """
        Return the scene graph plus derived counts for a single text.

        Keys: objects, attributes ([obj,attr]), relations ([o1,rel,o2]),
        method, n_attribute_tokens, n_relation_tokens, n_objects,
        total_words, semantic_density.
        """
        if text in self._cache:
            graph = self._cache[text]
        else:
            graph = None
            if self.use_llm and self._llm_ok is not False:
                graph = self._llm_extract(text)
                if graph is None and self._llm_ok is None:
                    logger.warning(
                        "Ollama unavailable for semantic extraction — "
                        "falling back to keyword counts for this run."
                    )
                    self._llm_ok = False
                elif graph is not None:
                    self._llm_ok = True
            if graph is None:
                graph = self._keyword_extract(text)
            self._cache[text] = graph

        n_attr = len(graph["attributes"])
        n_rel = len(graph["relations"])
        n_words = len(text.split())
        return {
            **graph,
            "n_objects": len(graph["objects"]),
            "n_attribute_tokens": n_attr,
            "n_relation_tokens": n_rel,
            "total_words": n_words,
            "semantic_density": (n_attr + n_rel) / n_words if n_words else 0.0,
        }

    # Backwards-compatible alias: old code calls .count()
    def count(self, text: str) -> dict:
        return self.extract(text)


# Backwards-compatible name so existing imports keep working.
SemanticCounter = SemanticExtractor

# Module-level default extractor (LLM on).
_DEFAULT_COUNTER = SemanticExtractor(use_llm=True)


def count_semantic_tokens(text: str, counter: Optional[SemanticExtractor] = None) -> dict:
    """
    Extract scene-graph structure + counts for a text string.

    Uses the LLM-backed SemanticExtractor by default (keyword fallback if the
    LLM is unavailable). Returns objects/attributes/relations/method plus
    n_attribute_tokens, n_relation_tokens, n_objects, total_words,
    semantic_density.
    """
    return (counter or _DEFAULT_COUNTER).extract(text)


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
    raw_stats = [counter.extract(p) for p in raw_prompts]
    rw_stats = [counter.extract(p) for p in rewritten_prompts]
    n = len(raw_prompts)

    def avg(lst, key):
        return sum(d[key] for d in lst) / n if n else 0.0

    # Fraction of rewrites that used the LLM (vs keyword fallback) — report
    # this so silent degradation to keyword counting is visible, not hidden.
    llm_frac = (
        sum(1 for d in rw_stats if d.get("method") == "llm") / n if n else 0.0
    )

    return {
        # counts (renamed from *_density — these are counts, not ratios)
        f"{pipeline_name}/raw_attr_count":       avg(raw_stats, "n_attribute_tokens"),
        f"{pipeline_name}/raw_rel_count":        avg(raw_stats, "n_relation_tokens"),
        f"{pipeline_name}/raw_obj_count":        avg(raw_stats, "n_objects"),
        f"{pipeline_name}/rw_attr_count":        avg(rw_stats, "n_attribute_tokens"),
        f"{pipeline_name}/rw_rel_count":         avg(rw_stats, "n_relation_tokens"),
        f"{pipeline_name}/rw_obj_count":         avg(rw_stats, "n_objects"),
        f"{pipeline_name}/attr_count_gain":      avg(rw_stats, "n_attribute_tokens") - avg(raw_stats, "n_attribute_tokens"),
        f"{pipeline_name}/rel_count_gain":       avg(rw_stats, "n_relation_tokens") - avg(raw_stats, "n_relation_tokens"),
        f"{pipeline_name}/obj_count_gain":       avg(rw_stats, "n_objects") - avg(raw_stats, "n_objects"),
        # true density (the ratio)
        f"{pipeline_name}/raw_semantic_density": avg(raw_stats, "semantic_density"),
        f"{pipeline_name}/rw_semantic_density":  avg(rw_stats, "semantic_density"),
        # extraction provenance
        f"{pipeline_name}/llm_extract_fraction": llm_frac,
    }


def compute_embedding_separation(
    embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """
    Compute pairwise cosine similarity statistics for a batch of embeddings.

    Pooling: uses the LAST non-padding token (the EOS position), which is the
    representation CLIP is actually trained to use as its global summary — not
    a mean over all positions (which dilutes signal with padding tokens). If an
    attention_mask is given, the EOS index is the last masked-in position per
    row; otherwise it falls back to the final sequence position.

    Parameters
    ----------
    embeddings : torch.Tensor
        Shape: (n_prompts, seq_len, hidden_dim)
    attention_mask : torch.Tensor, optional
        Shape: (n_prompts, seq_len); 1 for real tokens, 0 for padding.

    Returns
    -------
    dict with mean_pairwise_cosine_similarity, std_pairwise_cosine_similarity,
    mean_pairwise_distance (1 - similarity).
    """
    if attention_mask is not None:
        # EOS = last position where mask == 1, per row.
        lengths = attention_mask.sum(dim=1).long()                  # (n,)
        eos_idx = (lengths - 1).clamp(min=0)                        # (n,)
        pooled = embeddings[torch.arange(embeddings.size(0)), eos_idx]
    else:
        # No mask available — use the final token position.
        pooled = embeddings[:, -1, :]
    pooled = F.normalize(pooled, dim=-1)

    sim_matrix = torch.mm(pooled, pooled.T)           # (n, n)
    n = pooled.shape[0]
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

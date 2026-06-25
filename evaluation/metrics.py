"""
evaluation/metrics.py

Core evaluation metrics for the 2×3 experimental design.

Metrics
-------
- CLIPScore          : prompt–image semantic alignment (RQ1, RQ2, RQ4)
- FID                : perceptual realism vs. reference distribution (RQ2, RQ4)
- AttributeBindingScorer : correct colour/attribute–object assignment (RQ2, RQ4)
- RelationAccuracyScorer : spatial/relational constraint satisfaction (RQ2, RQ4)

Attribute binding and relation accuracy use BLIP-2 as a VQA backbone.
Questions are templated from parsed prompt structure.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLIPScore
# ---------------------------------------------------------------------------

class CLIPScorer:
    """
    Computes CLIPScore (prompt–image cosine similarity in CLIP space).

    Uses torchmetrics.multimodal.CLIPScore when available, falling back
    to a manual cosine similarity implementation.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace CLIP model to use for scoring. Should match the encoder
        used for conditioning so scores are comparable.
    device : str
    """

    def __init__(
        self,
        model_name_or_path: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor
        logger.info("Loading CLIP for CLIPScore from %s", self.model_name_or_path)
        self._processor = CLIPProcessor.from_pretrained(self.model_name_or_path)
        self._model = CLIPModel.from_pretrained(
            self.model_name_or_path, torch_dtype=torch.float16
        ).to(self.device).eval()

    @torch.no_grad()
    def score(self, image: Image.Image, prompt: str) -> float:
        """Compute CLIPScore for a single image–prompt pair. Returns [0, 1]."""
        self._load()
        inputs = self._processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        outputs = self._model(**inputs)
        # Cosine similarity of L2-normalised embeddings
        img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        sim = (img_emb * txt_emb).sum(dim=-1).item()
        return float(sim)

    def score_batch(
        self, images: list[Image.Image], prompts: list[str]
    ) -> list[float]:
        return [self.score(img, p) for img, p in zip(images, prompts)]

    def mean_score(
        self, images: list[Image.Image], prompts: list[str]
    ) -> float:
        scores = self.score_batch(images, prompts)
        return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

class FIDScorer:
    """
    Computes Fréchet Inception Distance between real and generated image sets.

    Requires a reference image set (e.g. MS-COCO validation images matching
    the prompt topics). FID is computed using Inception-v3 features.

    Parameters
    ----------
    device : str
    feature_dims : int
        Inception feature dimensionality. 2048 is standard.
    """

    def __init__(
        self,
        device: str = "cuda",
        feature_dims: int = 2048,
    ) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.feature_dims = feature_dims
        self._fid = None

    def _load(self) -> None:
        if self._fid is not None:
            return
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
        except ImportError:
            raise ImportError(
                "torchmetrics is required for FID. "
                "Install with: pip install torchmetrics[image]"
            )
        self._fid = FrechetInceptionDistance(
            feature=self.feature_dims, normalize=True
        ).to(self.device)

    def _to_uint8_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to (1, 3, H, W) uint8 tensor."""
        import torchvision.transforms.functional as TF
        img = image.convert("RGB").resize((299, 299), Image.BILINEAR)
        t = TF.to_tensor(img)          # float [0,1], (3, H, W)
        t = (t * 255).byte()           # uint8
        return t.unsqueeze(0)          # (1, 3, H, W)

    def update_real(self, images: list[Image.Image]) -> None:
        """Feed reference (real) images."""
        self._load()
        for img in images:
            t = self._to_uint8_tensor(img).to(self.device)
            self._fid.update(t, real=True)

    def update_generated(self, images: list[Image.Image]) -> None:
        """Feed generated images."""
        self._load()
        for img in images:
            t = self._to_uint8_tensor(img).to(self.device)
            self._fid.update(t, real=False)

    def compute(self) -> float:
        """Return FID score. Lower is better."""
        self._load()
        return float(self._fid.compute().item())

    def reset(self) -> None:
        if self._fid is not None:
            self._fid.reset()


# ---------------------------------------------------------------------------
# Prompt parser (shared by attribute binding + relation scorers)
# ---------------------------------------------------------------------------

# Simple colour vocabulary — extend as needed
# Colour attributes (T2I-CompBench++ color_binding)
_COLOURS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "white", "black", "grey", "gray", "brown", "gold", "silver",
    "beige", "cyan", "magenta", "teal", "navy", "maroon", "olive",
    "turquoise", "violet", "indigo", "crimson", "golden",
]

# Shape attributes (T2I-CompBench++ shape_binding)
_SHAPES = [
    "oblong", "teardrop", "pentagonal", "spherical", "cubic", "cylindrical",
    "pyramidal", "diamond", "oval", "rectangular", "triangular", "conical",
    "round", "square", "circular",
]

# Size / general visual attributes that appear across all binding categories
_VISUAL_ATTRS = [
    "striped", "spotted", "fluffy", "curly", "long", "short", "tall",
    "small", "large", "tiny", "big",
]

# Texture/material attributes (T2I-CompBench++ texture_binding)
_TEXTURES = [
    "wooden", "metallic", "metal", "glass", "ceramic", "velvet", "leather",
    "plastic", "rubber", "fabric", "fluffy",
]

# Combined list used by parse_attribute_pairs — sorted longest-first so
# multi-word attributes (none currently) would match before single tokens.
_ALL_ATTRIBUTES = sorted(
    set(_COLOURS + _SHAPES + _VISUAL_ATTRS + _TEXTURES),
    key=lambda x: -len(x),
)

_SPATIAL_RELATIONS = [
    # T2I-CompBench++ phrasing (on the X of / on side of)
    "on the left of", "on the right of",
    "on the top of", "on the bottom of",
    "on side of",
    # Standard English phrasing
    "to the left of", "to the right of",
    "above", "below",
    "in front of", "behind",
    "on top of", "under", "beside",
    "next to", "near", "between", "inside", "on",
]


def parse_attribute_pairs(prompt: str) -> list[tuple[str, str]]:
    """
    Extract (attribute, object) pairs from a prompt using regex heuristics.

    Returns a list of (colour_or_attr, noun) tuples, e.g.:
        "a red cat beside a blue dog"
        → [("red", "cat"), ("blue", "dog")]
    """
    pairs = []
    attr_pattern = "|".join(re.escape(c) for c in _ALL_ATTRIBUTES)
    # Match "a/an <attr> <noun>" or just "<attr> <noun>"
    pattern = rf'\b({attr_pattern})\s+(\w+)\b'
    for match in re.finditer(pattern, prompt.lower()):
        attr, noun = match.group(1), match.group(2)
        # Filter out stop words that can follow an attribute token
        if noun not in {"and", "or", "on", "in", "at", "of", "the", "a", "an"}:
            pairs.append((attr, noun))
    return pairs


def parse_spatial_relations(prompt: str) -> list[tuple[str, str, str]]:
    """
    Extract (object_a, relation, object_b) triples from a prompt.

    Returns list of (subj, relation, obj) tuples, e.g.:
        "a dog to the left of a cat"
        → [("dog", "to the left of", "cat")]
    """
    _ARTICLES = {"a", "an", "the"}
    triples = []
    p = prompt.lower()
    # Sort longest-first so "on the right of" is matched before bare "on"
    consumed: set[int] = set()
    for rel in sorted(_SPATIAL_RELATIONS, key=lambda r: -len(r)):
        # Use word-boundary anchors on the first and last token of the relation
        pattern = rf'\b{re.escape(rel)}\b'
        m = re.search(pattern, p)
        if m is None:
            continue
        idx = m.start()
        # Skip if this position was already claimed by a longer relation
        if idx in consumed:
            continue
        consumed.update(range(idx, idx + len(rel)))
        # Subject: last non-article word before the relation
        before = p[:idx].strip().split()
        subj = ""
        for w in reversed(before):
            w = w.strip(".,;")
            if w and w not in _ARTICLES:
                subj = w
                break
        # Object: first non-article word after the relation
        after = p[m.end():].strip().split()
        obj = ""
        for w in after:
            w = w.strip(".,;")
            if w and w not in _ARTICLES:
                obj = w
                break
        if subj and obj:
            triples.append((subj, rel, obj))
    return triples


# ---------------------------------------------------------------------------
# Attribute Binding Accuracy (VQA-based)
# ---------------------------------------------------------------------------

class AttributeBindingScorer:
    """
    Scores attribute–object binding accuracy using BLIP-2 VQA.

    For each (attribute, object) pair parsed from the prompt, asks:
        "What colour is the <object>?" → checks if answer matches attribute.
    Or more generally:
        "Is there a <attribute> <object> in this image?" → yes/no.

    The yes/no question is more reliable for colour attributes.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace BLIP-2 model checkpoint.
    device : str
    """

    def __init__(
        self,
        model_name_or_path: str = "Salesforce/blip2-flan-t5-xl",
        device: str = "cuda",
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        logger.info("Loading BLIP-2 from %s for VQA scoring", self.model_name_or_path)
        self._processor = Blip2Processor.from_pretrained(self.model_name_or_path)
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16,
        ).to(self.device).eval()

    @torch.no_grad()
    def _vqa(self, image: Image.Image, question: str) -> str:
        self._load()
        inputs = self._processor(images=image, text=question, return_tensors="pt").to(
            self.device, dtype=torch.float16
        )
        out = self._model.generate(**inputs, max_new_tokens=20)
        return self._processor.decode(out[0], skip_special_tokens=True).strip().lower()

    def score(self, image: Image.Image, prompt: str) -> dict:
        """
        Score attribute binding for a single image–prompt pair.

        Returns
        -------
        dict with keys:
            n_pairs : int — number of attribute–object pairs in prompt
            n_correct : int — pairs where VQA answer matched the attribute
            accuracy : float — n_correct / n_pairs (0.0 if no pairs)
            details : list[dict] — per-pair breakdown
        """
        pairs = parse_attribute_pairs(prompt)
        if not pairs:
            return {"n_pairs": 0, "n_correct": 0, "accuracy": 0.0, "details": []}

        n_correct = 0
        details = []
        for attr, obj in pairs:
            question = f"Is there a {attr} {obj} in this image? Answer yes or no."
            answer = self._vqa(image, question)
            correct = answer.startswith("yes")
            n_correct += int(correct)
            details.append({"attr": attr, "obj": obj, "answer": answer, "correct": correct})

        return {
            "n_pairs": len(pairs),
            "n_correct": n_correct,
            "accuracy": n_correct / len(pairs),
            "details": details,
        }

    def mean_accuracy(self, images: list[Image.Image], prompts: list[str]) -> float:
        scores = [self.score(img, p)["accuracy"] for img, p in zip(images, prompts)]
        return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Relation Accuracy (VQA-based)
# ---------------------------------------------------------------------------

class RelationAccuracyScorer:
    """
    Scores spatial/relational constraint satisfaction using BLIP-2 VQA.

    For each (subj, relation, obj) triple parsed from the prompt, asks:
        "Is the <subj> <relation> the <obj>? Answer yes or no."

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace BLIP-2 model checkpoint (shared with AttributeBindingScorer
        if both are used in the same experiment — pass the same loaded model
        to avoid double VRAM usage).
    device : str
    """

    def __init__(
        self,
        model_name_or_path: str = "Salesforce/blip2-flan-t5-xl",
        device: str = "cuda",
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        logger.info("Loading BLIP-2 for relation accuracy scoring")
        self._processor = Blip2Processor.from_pretrained(self.model_name_or_path)
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16,
        ).to(self.device).eval()

    @torch.no_grad()
    def _vqa(self, image: Image.Image, question: str) -> str:
        self._load()
        inputs = self._processor(images=image, text=question, return_tensors="pt").to(
            self.device, dtype=torch.float16
        )
        out = self._model.generate(**inputs, max_new_tokens=20)
        return self._processor.decode(out[0], skip_special_tokens=True).strip().lower()

    def score(self, image: Image.Image, prompt: str) -> dict:
        """
        Score relational accuracy for a single image–prompt pair.

        Returns dict with n_relations, n_correct, accuracy, details.
        """
        triples = parse_spatial_relations(prompt)
        if not triples:
            return {"n_relations": 0, "n_correct": 0, "accuracy": 0.0, "details": []}

        n_correct = 0
        details = []
        for subj, rel, obj in triples:
            question = f"Is the {subj} {rel} the {obj}? Answer yes or no."
            answer = self._vqa(image, question)
            correct = answer.startswith("yes")
            n_correct += int(correct)
            details.append({
                "subj": subj, "relation": rel, "obj": obj,
                "answer": answer, "correct": correct,
            })

        return {
            "n_relations": len(triples),
            "n_correct": n_correct,
            "accuracy": n_correct / len(triples),
            "details": details,
        }

    def mean_accuracy(self, images: list[Image.Image], prompts: list[str]) -> float:
        scores = [self.score(img, p)["accuracy"] for img, p in zip(images, prompts)]
        return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Convenience: score all metrics at once
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    pipeline_name: str
    clip_score: float
    attr_binding_accuracy: float
    relation_accuracy: float
    fid: Optional[float] = None  # set after FID computation over full set


def score_all(
    pipeline_name: str,
    images: list[Image.Image],
    prompts: list[str],
    clip_scorer: CLIPScorer,
    attr_scorer: AttributeBindingScorer,
    rel_scorer: RelationAccuracyScorer,
) -> EvalResult:
    """Run CLIPScore, attribute binding, and relation accuracy for a batch."""
    return EvalResult(
        pipeline_name=pipeline_name,
        clip_score=clip_scorer.mean_score(images, prompts),
        attr_binding_accuracy=attr_scorer.mean_accuracy(images, prompts),
        relation_accuracy=rel_scorer.mean_accuracy(images, prompts),
    )

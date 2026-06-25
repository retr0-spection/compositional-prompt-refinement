"""
encoders/clip_encoder.py

Standard CLIP text encoder (77-token context window).

SD 2.1 uses laion/CLIP-ViT-H-14-laion2B-s32B-b79K (hidden_dim=1024).
Override model_id in CLIPEncoderConfig if you need a different checkpoint
(e.g. openai/clip-vit-large-patch14 for hidden_dim=768).

Output shape: (1, 77, hidden_dim) — ready for cross-attention injection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import CLIPTextModel, CLIPTokenizer

from encoders.base import TextEncoder

logger = logging.getLogger(__name__)

_CONTEXT_LENGTH = 77


def _resolve_device(device: str) -> str:
    """Fall back to CPU if CUDA is requested but unavailable."""
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU.")
        return "cpu"
    return device


@dataclass
class CLIPEncoderConfig:
    # SD 2.1 uses the H/14 variant; swap for L/14 if needed
    model_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.float16


class CLIPEncoder(TextEncoder):
    """
    Standard CLIP text encoder — 77-token context window.

    The model is loaded lazily on the first call to encode() or token_count()
    so that importing this module does not trigger a weight download.

    Usage
    -----
    encoder = CLIPEncoder()
    embedding = encoder.encode("a red cat beside a blue dog")
    print(embedding.shape)  # torch.Size([1, 77, 1024])
    """

    def __init__(self, config: Optional[CLIPEncoderConfig] = None) -> None:
        self._config = config or CLIPEncoderConfig()
        self._model: Optional[CLIPTextModel] = None
        self._tokenizer: Optional[CLIPTokenizer] = None

    @property
    def name(self) -> str:
        return "clip"

    @property
    def context_length(self) -> int:
        return _CONTEXT_LENGTH

    def _load(self) -> None:
        if self._model is not None:
            return

        cfg = self._config
        device = _resolve_device(cfg.device)

        logger.info("Loading CLIP tokenizer from %s", cfg.model_id)
        self._tokenizer = CLIPTokenizer.from_pretrained(cfg.model_id)

        logger.info("Loading CLIP text model from %s (dtype=%s, device=%s)",
                    cfg.model_id, cfg.torch_dtype, device)
        self._model = (
            CLIPTextModel.from_pretrained(cfg.model_id, torch_dtype=cfg.torch_dtype)
            .to(device)
            .eval()
        )
        # Store the resolved device so encode() can use it
        self._device = device
        logger.info("CLIP encoder loaded.")

    def token_count(self, text: str) -> int:
        self._load()
        tokens = self._tokenizer(text, add_special_tokens=True, return_tensors="pt")
        return int(tokens["input_ids"].shape[1])

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text into a CLIP conditioning embedding.

        Prompts longer than 77 tokens are truncated. Call token_count() first
        if you need to detect truncation before encoding.

        Returns
        -------
        torch.Tensor
            Shape: (1, 77, hidden_dim)
        """
        self._load()
        encoded = self._tokenizer(
            text,
            padding="max_length",
            max_length=_CONTEXT_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)

        outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state: (1, seq_len, hidden_dim)
        return outputs.last_hidden_state

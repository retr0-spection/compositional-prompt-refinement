"""
encoders/longclip_encoder.py

Long-CLIP text encoder — 248-token context window.

Long-CLIP (Zhang et al., 2024) extends CLIP's position embeddings from 77
to 248 tokens via interpolation, enabling encoding of longer, richer prompts
without retraining the image backbone.

Checkpoint: longclip-L.pt — downloaded automatically from HuggingFace Hub
(tries several candidate repo IDs). If auto-download fails, set
longclip_checkpoint in experiment.yaml to a local path. See the error
message for a manual Google Drive download link.

Output shape: (1, 248, hidden_dim) — ready for cross-attention injection.

Note on SD 2.1 compatibility
----------------------------
SD 2.1's UNet cross-attention expects seq_len=77 from its CLIP-H encoder.
When substituting Long-CLIP-L (hidden_dim=768, seq_len=248), you may need
to either:
  a) Truncate the embedding to 77 tokens (loses long-context benefit), or
  b) Fine-tune the cross-attention projections to accept seq_len=248.
This encoder returns the full (1, 248, 768) tensor; the T2I runner is
responsible for any necessary adaptation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

from encoders.base import TextEncoder

logger = logging.getLogger(__name__)

_CONTEXT_LENGTH = 248
_BASE_CONTEXT_LENGTH = 77   # original CLIP position embedding size
# Candidate (repo_id, filename) pairs tried in order.
_HF_CANDIDATES = [
    ("BeichenZhang/LongCLIP-L", "longclip-L.pt"),   # correct repo (no hyphen)
    ("BeichenZhang/Long-CLIP-L", "longclip-L.pt"),  # common misspelling
    ("BeichenZhang/Long-CLIP",   "longclip-L.pt"),
]
_MANUAL_DOWNLOAD_MSG = (
    "Could not download the Long-CLIP-L checkpoint automatically.\n"
    "Download it manually from:\n"
    "  https://huggingface.co/BeichenZhang/LongCLIP-L/blob/main/longclip-L.pt\n"
    "Then set  longclip_checkpoint: /path/to/longclip-L.pt  in config/experiment.yaml\n"
    "or pass   checkpoint_path='/path/to/longclip-L.pt'  to LongCLIPEncoderConfig."
)


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU.")
        return "cpu"
    return device


def _extend_position_embeddings(
    model: CLIPTextModel,
    new_length: int = _CONTEXT_LENGTH,
) -> CLIPTextModel:
    """
    Interpolate CLIP's position embeddings from 77 to new_length tokens.

    Follows the Long-CLIP paper: positional embeddings are split into a
    front portion (token positions 0..19) kept as-is, and a tail portion
    (positions 20..76) that is interpolated to cover positions 20..new_length-1.
    This preserves short-range positional structure while expanding capacity.

    Reference: Zhang et al. (2024), Section 3.1.
    """
    text_model = model.text_model
    old_embed: nn.Embedding = text_model.embeddings.position_embedding
    old_weight = old_embed.weight.data.float()   # (77, hidden_dim)
    hidden_dim = old_weight.shape[1]

    # Split: positions 0..19 stay, 20..76 are interpolated
    front = old_weight[:20]                      # (20, hidden_dim)
    tail = old_weight[20:]                       # (57, hidden_dim)

    new_tail_len = new_length - 20
    # Interpolate tail: (1, 57, hidden_dim) -> (1, new_tail_len, hidden_dim)
    tail_interp = torch.nn.functional.interpolate(
        tail.unsqueeze(0).transpose(1, 2),       # (1, hidden_dim, 57)
        size=new_tail_len,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2).squeeze(0)                 # (new_tail_len, hidden_dim)

    new_weight = torch.cat([front, tail_interp], dim=0)  # (new_length, hidden_dim)

    new_embed = nn.Embedding(new_length, hidden_dim)
    new_embed.weight = nn.Parameter(new_weight.to(old_embed.weight.dtype))
    text_model.embeddings.position_embedding = new_embed

    # Update position_ids buffer
    text_model.embeddings.position_ids = (
        torch.arange(new_length).unsqueeze(0)
    )

    # Update config so the model knows its new max positions
    model.config.max_position_embeddings = new_length
    text_model.config.max_position_embeddings = new_length

    return model


@dataclass
class LongCLIPEncoderConfig:
    # Path to a local longclip-L.pt checkpoint, or leave as None to
    # download automatically from BeichenZhang/Long-CLIP-L on HF Hub.
    checkpoint_path: Optional[str] = None
    # Base CLIP architecture to load before applying Long-CLIP weights
    base_model_id: str = "openai/clip-vit-large-patch14"
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.float16


class LongCLIPEncoder(TextEncoder):
    """
    Long-CLIP text encoder — 248-token context window.

    Loads openai/clip-vit-large-patch14 from transformers, extends its
    position embeddings to 248 tokens via interpolation, then overlays the
    Long-CLIP checkpoint weights (which already encode this extension).

    If no checkpoint_path is provided in the config, the checkpoint is
    downloaded from BeichenZhang/Long-CLIP-L on the Hugging Face Hub.

    Usage
    -----
    encoder = LongCLIPEncoder()
    embedding = encoder.encode("a detailed scene description ...")
    print(embedding.shape)  # torch.Size([1, 248, 768])
    """

    def __init__(self, config: Optional[LongCLIPEncoderConfig] = None) -> None:
        self._config = config or LongCLIPEncoderConfig()
        self._model: Optional[CLIPTextModel] = None
        self._tokenizer: Optional[CLIPTokenizer] = None
        self._device: Optional[str] = None

    @property
    def name(self) -> str:
        return "longclip"

    @property
    def context_length(self) -> int:
        return _CONTEXT_LENGTH

    def _get_checkpoint_path(self) -> str:
        cfg = self._config
        if cfg.checkpoint_path and Path(cfg.checkpoint_path).exists():
            return cfg.checkpoint_path

        # Try to download from HF Hub — attempt each candidate in order
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required to download the Long-CLIP checkpoint. "
                "Install it with: pip install huggingface_hub"
            ) from exc

        last_exc: Exception | None = None
        for repo_id, filename in _HF_CANDIDATES:
            try:
                logger.info("Trying Long-CLIP checkpoint: %s/%s", repo_id, filename)
                return hf_hub_download(repo_id=repo_id, filename=filename)
            except Exception as exc:
                logger.debug("  → failed (%s)", exc)
                last_exc = exc

        raise RuntimeError(
            f"{_MANUAL_DOWNLOAD_MSG}\n\n(Last error: {last_exc})"
        )

    def _load(self) -> None:
        if self._model is not None:
            return

        cfg = self._config
        device = _resolve_device(cfg.device)

        # 1. Load base CLIP architecture + tokenizer
        logger.info("Loading Long-CLIP tokenizer from %s", cfg.base_model_id)
        tokenizer = CLIPTokenizer.from_pretrained(cfg.base_model_id)
        # Override max_length so tokenizer doesn't truncate at 77
        tokenizer.model_max_length = _CONTEXT_LENGTH

        logger.info("Loading base CLIP text model from %s", cfg.base_model_id)
        model = CLIPTextModel.from_pretrained(
            cfg.base_model_id, torch_dtype=torch.float32  # load in fp32 for safe interpolation
        )

        # 2. Extend position embeddings 77 → 248
        logger.info("Extending position embeddings to %d tokens", _CONTEXT_LENGTH)
        model = _extend_position_embeddings(model, new_length=_CONTEXT_LENGTH)

        # 3. Load Long-CLIP checkpoint and overlay text model weights
        ckpt_path = self._get_checkpoint_path()
        logger.info("Loading Long-CLIP checkpoint from %s", ckpt_path)
        state_dict = torch.load(ckpt_path, map_location="cpu")

        # The checkpoint may be wrapped under a 'model' key
        if "model" in state_dict:
            state_dict = state_dict["model"]

        # Extract text encoder weights only
        text_prefix = "transformer.text_model."
        clip_prefix = "text_model."
        text_weights = {}
        for k, v in state_dict.items():
            if k.startswith(text_prefix):
                text_weights[k[len(text_prefix):]] = v
            elif k.startswith(clip_prefix):
                text_weights[k[len(clip_prefix):]] = v

        if text_weights:
            missing, unexpected = model.text_model.load_state_dict(text_weights, strict=False)
            if missing:
                logger.debug("Long-CLIP: missing text keys: %s", missing)
            if unexpected:
                logger.debug("Long-CLIP: unexpected text keys: %s", unexpected)
        else:
            logger.warning(
                "Could not extract text model weights from checkpoint. "
                "Keys found: %s", list(state_dict.keys())[:10]
            )

        # 4. Cast to target dtype and move to device
        model = model.to(dtype=cfg.torch_dtype).to(device).eval()

        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        logger.info("Long-CLIP encoder loaded (context_length=%d).", _CONTEXT_LENGTH)

    def token_count(self, text: str) -> int:
        self._load()
        tokens = self._tokenizer(text, add_special_tokens=True, return_tensors="pt")
        return int(tokens["input_ids"].shape[1])

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text into a Long-CLIP conditioning embedding.

        Prompts longer than 248 tokens are truncated. This is far less likely
        than with standard CLIP (77 tokens) but still possible for very long
        LLaDA expansions.

        Returns
        -------
        torch.Tensor
            Shape: (1, 248, 768)
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
        return outputs.last_hidden_state  # (1, 248, 768)

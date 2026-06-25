"""
generation/t2i_runner.py

Stable Diffusion 2.1 wrapper that accepts a pre-computed conditioning
embedding (prompt_embeds) rather than a raw text string.

This is the key integration point between the conditioning pipeline and
the frozen T2I backbone. The SD 2.1 UNet is never fine-tuned; all variation
is in the conditioning tensor passed via prompt_embeds.

Architecture note
-----------------
SD 2.1 uses CLIP-H (hidden_dim=1024, seq_len=77). If you pass a LongCLIP
embedding (hidden_dim=768, seq_len=248), the cross-attention projection
will raise a dimension mismatch. Two options:

  1. Set `project_embeds=True` in T2IRunnerConfig — a learned linear layer
     projects (seq_len, 768) → (77, 1024) before passing to the UNet.
     (This requires fine-tuning the projection; see notes below.)

  2. Truncate LongCLIP output to seq_len=77 and project 768→1024.
     Simpler but defeats the long-context benefit.

For the research proposal's scope, the default is to use CLIP-H throughout
and rely on the rewriter (LLaDA / Ollama) to produce better short embeddings.
The LongCLIP path is provided for ablation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU. Generation will be slow.")
        return "cpu"
    return device


# ---------------------------------------------------------------------------
# Optional embedding projection (LongCLIP → SD 2.1 cross-attention dim)
# ---------------------------------------------------------------------------

class EmbeddingProjector(nn.Module):
    """
    Project (batch, src_seq, src_dim) → (batch, tgt_seq, tgt_dim).

    Used when LongCLIP embeddings (seq=248, dim=768) need to be fed to an
    SD 2.1 UNet that expects (seq=77, dim=1024).

    The projection is a two-step linear: first project dim, then interpolate
    sequence length. This module is only used for ablation — it starts
    untrained (random weights) unless a checkpoint is provided.
    """

    def __init__(
        self,
        src_dim: int = 768,
        tgt_dim: int = 1024,
        src_seq: int = 248,
        tgt_seq: int = 77,
    ) -> None:
        super().__init__()
        self.tgt_seq = tgt_seq
        self.dim_proj = nn.Linear(src_dim, tgt_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, src_seq, src_dim)
        x = self.dim_proj(x)                          # → (batch, src_seq, tgt_dim)
        x = x.permute(0, 2, 1)                        # → (batch, tgt_dim, src_seq)
        x = torch.nn.functional.interpolate(
            x, size=self.tgt_seq, mode="linear", align_corners=False
        )                                              # → (batch, tgt_dim, tgt_seq)
        return x.permute(0, 2, 1)                     # → (batch, tgt_seq, tgt_dim)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class T2IRunnerConfig:
    model_id: str = "stabilityai/stable-diffusion-2-1"
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.float16

    # Sampling defaults (overridable per call)
    num_inference_steps: int = 50
    default_cfg_scale: float = 7.5
    image_height: int = 512
    image_width: int = 512

    # Set True if passing LongCLIP embeddings (requires projection layer)
    project_embeds: bool = False
    projector_checkpoint: Optional[str] = None  # path to trained projector weights

    # Source / target dims for the projector
    src_dim: int = 768
    tgt_dim: int = 1024
    src_seq: int = 248
    tgt_seq: int = 77


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class T2IRunner:
    """
    Wrapper around StableDiffusionPipeline that accepts prompt_embeds.

    The SD model is loaded lazily on first generate() call to allow
    LLaDA to be unloaded first (both are large; VRAM is limited).

    Usage
    -----
    runner = T2IRunner()
    image = runner.generate(prompt_embeds=embedding, cfg_scale=7.5, seed=42)
    image.save("output.png")
    """

    def __init__(self, config: Optional[T2IRunnerConfig] = None) -> None:
        self.config = config or T2IRunnerConfig()
        self._pipe = None
        self._projector: Optional[EmbeddingProjector] = None
        self._device: Optional[str] = None
        self._cached_neg_embeds: Optional[torch.Tensor] = None  # empty-string embed, computed once

    def _load(self) -> None:
        if self._pipe is not None:
            return

        cfg = self.config
        device = _resolve_device(cfg.device)
        self._device = device

        logger.info("Loading SD 2.1 pipeline from %s", cfg.model_id)
        from diffusers import StableDiffusionPipeline

        self._pipe = StableDiffusionPipeline.from_pretrained(
            cfg.model_id,
            torch_dtype=cfg.torch_dtype,
        ).to(device)

        # Disable the internal safety checker for research use
        self._pipe.safety_checker = None
        self._pipe.requires_safety_checker = False

        if cfg.project_embeds:
            logger.info(
                "Initialising EmbeddingProjector (%d→%d dim, %d→%d seq)",
                cfg.src_dim, cfg.tgt_dim, cfg.src_seq, cfg.tgt_seq,
            )
            self._projector = EmbeddingProjector(
                src_dim=cfg.src_dim,
                tgt_dim=cfg.tgt_dim,
                src_seq=cfg.src_seq,
                tgt_seq=cfg.tgt_seq,
            ).to(device)

            if cfg.projector_checkpoint:
                ckpt = torch.load(cfg.projector_checkpoint, map_location=device)
                self._projector.load_state_dict(ckpt)
                logger.info("Loaded projector weights from %s", cfg.projector_checkpoint)
            else:
                logger.warning(
                    "EmbeddingProjector has random weights — "
                    "pass projector_checkpoint or fine-tune before using LongCLIP."
                )

        logger.info("SD 2.1 loaded on %s.", device)

    def _prepare_embeds(
        self, prompt_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Move embeds to the correct device/dtype and optionally project them.
        """
        cfg = self.config
        embeds = prompt_embeds.to(self._device, dtype=cfg.torch_dtype)
        if self._projector is not None:
            embeds = self._projector(embeds)
        return embeds

    def _negative_embeds(self) -> torch.Tensor:
        """
        Return the unconditional (empty-string) embedding for CFG.

        Result is cached after the first call — the empty-string encoding is
        identical for every prompt, so there is no need to recompute it.
        """
        if self._cached_neg_embeds is not None:
            return self._cached_neg_embeds

        self._load()
        with torch.no_grad():
            uncond = self._pipe.encode_prompt(
                prompt="",
                device=self._device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=None,
            )
        # encode_prompt returns (prompt_embeds, negative_prompt_embeds)
        self._cached_neg_embeds = uncond[1]
        return self._cached_neg_embeds

    @torch.no_grad()
    def generate(
        self,
        prompt_embeds: torch.Tensor,
        cfg_scale: Optional[float] = None,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> Image.Image:
        """
        Generate one image from a pre-computed conditioning embedding.

        Parameters
        ----------
        prompt_embeds : torch.Tensor
            Shape: (1, seq_len, hidden_dim). From any ConditioningPipeline.
        cfg_scale : float | None
            Guidance scale. Defaults to config.default_cfg_scale.
        seed : int | None
            If provided, used to seed the latent noise generator for
            reproducibility. Combine with utils.seed.set_seed() for full
            global reproducibility.
        num_inference_steps : int | None
            Denoising steps. Defaults to config.num_inference_steps.
        negative_prompt_embeds : torch.Tensor | None
            Unconditional embeds for CFG. If None, generated from empty string.

        Returns
        -------
        PIL.Image.Image
        """
        self._load()
        cfg = self.config

        scale = cfg_scale if cfg_scale is not None else cfg.default_cfg_scale
        steps = num_inference_steps or cfg.num_inference_steps

        embeds = self._prepare_embeds(prompt_embeds)

        if negative_prompt_embeds is None:
            neg_embeds = self._negative_embeds().to(self._device, dtype=cfg.torch_dtype)
        else:
            neg_embeds = self._prepare_embeds(negative_prompt_embeds)

        generator = None
        if seed is not None:
            from utils.seed import get_generator
            generator = get_generator(seed, device=self._device)

        output = self._pipe(
            prompt_embeds=embeds,
            negative_prompt_embeds=neg_embeds,
            guidance_scale=scale,
            num_inference_steps=steps,
            height=cfg.image_height,
            width=cfg.image_width,
            generator=generator,
        )
        return output.images[0]

    def generate_batch(
        self,
        prompt_embeds_list: list[torch.Tensor],
        cfg_scale: Optional[float] = None,
        seeds: Optional[list[int]] = None,
        num_inference_steps: Optional[int] = None,
    ) -> list[Image.Image]:
        """
        Generate images for a list of pre-computed embeddings.

        Runs sequentially to avoid OOM on single-GPU setups.
        Each prompt uses its own seed if `seeds` is provided.
        """
        seeds = seeds or [None] * len(prompt_embeds_list)
        return [
            self.generate(
                prompt_embeds=emb,
                cfg_scale=cfg_scale,
                seed=s,
                num_inference_steps=num_inference_steps,
            )
            for emb, s in zip(prompt_embeds_list, seeds)
        ]

    def unload(self) -> None:
        """Release GPU memory. Call between phases if VRAM is tight."""
        self._pipe = None
        self._projector = None
        self._cached_neg_embeds = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("T2IRunner unloaded.")

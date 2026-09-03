"""
generation/t2i_runner_v2.py

Backbone-agnostic T2I runner. Takes RAW TEXT (not pre-computed embeddings)
and owns all encoding internally, branching on the configured backbone:

  - sd21 : StableDiffusionPipeline,   single CLIP-H encoder
  - sdxl : StableDiffusionXLPipeline, native dual encoders (bigG + L) + pooled

This is the Option-2 design: each backbone uses its NATIVE conditioning, so
the encoder is no longer an experimental axis. Refinement (raw / ar / llada)
is the only independent variable; the backbone is a config switch.

    t2i:
      backbone: sdxl                                     # or sd21
      model_id: stabilityai/stable-diffusion-xl-base-1.0
      resolution: 1024
      prediction_type: epsilon                           # v_prediction for sd21

Why text-in: the pipelines (raw/ar/llada) produce rewritten TEXT; the runner
encodes it with whatever the backbone dictates. This keeps refinement logic
backbone-independent and avoids reimplementing SDXL's dual-encoder concat by
hand.

This file is parallel to the original t2i_runner.py (embedding-in, SD2.1 only),
which is kept working until this one is validated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU (very slow).")
        return "cpu"
    return device


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class T2IRunnerV2Config:
    # Backbone selection.
    backbone: str = "sd21"           # "sd21" | "sdxl"
    model_id: str = "sd2-community/stable-diffusion-2-1"
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.float16

    # Sampling.
    num_inference_steps: int = 50
    default_cfg_scale: float = 7.5
    resolution: int = 768            # 768 for sd21, 1024 for sdxl

    # Scheduler.
    prediction_type: str = "v_prediction"   # "epsilon" for sdxl base
    use_karras_sigmas: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "T2IRunnerV2Config":
        """
        Build from the experiment.yaml `t2i:` block, applying sensible
        backbone-specific defaults so a minimal config just works.
        """
        t2i = dict(d.get("t2i", d))  # accept either the block or a flat dict
        backbone = t2i.get("backbone", "sd21").lower()

        # Backbone-specific defaults; explicit config still overrides.
        defaults = {
            "sd21": dict(
                model_id="sd2-community/stable-diffusion-2-1",
                resolution=768,
                prediction_type="v_prediction",
            ),
            "sdxl": dict(
                model_id="stabilityai/stable-diffusion-xl-base-1.0",
                resolution=1024,
                prediction_type="epsilon",
            ),
        }[backbone]

        dtype = t2i.get("torch_dtype", torch.float16)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        # Backbone-coupled fields (model_id, resolution, prediction_type) must
        # be consistent with `backbone`. Only honour an explicit override of
        # these if it is NOT the other backbone's committed default — otherwise
        # a stale SDXL model_id in the yaml would load into an sd21 pipeline.
        _other = "sdxl" if backbone == "sd21" else "sd21"
        _other_defaults = {
            "sd21": dict(model_id="sd2-community/stable-diffusion-2-1",
                         resolution=768, prediction_type="v_prediction"),
            "sdxl": dict(model_id="stabilityai/stable-diffusion-xl-base-1.0",
                         resolution=1024, prediction_type="epsilon"),
        }[_other]

        def _coupled(key):
            val = t2i.get(key, defaults[key])
            # If the value matches the OTHER backbone's default, it's stale
            # config left over from a backbone switch — use this backbone's.
            if val == _other_defaults[key]:
                logger.warning(
                    "t2i.%s=%r matches the %s default but backbone=%s — "
                    "using %s's default instead to avoid a mismatch.",
                    key, val, _other, backbone, backbone)
                return defaults[key]
            return val

        return cls(
            backbone=backbone,
            model_id=_coupled("model_id"),
            device=t2i.get("device", "cuda"),
            torch_dtype=dtype,
            num_inference_steps=t2i.get("num_inference_steps", 50),
            default_cfg_scale=t2i.get("default_cfg_scale", 7.5),
            resolution=_coupled("resolution"),
            prediction_type=_coupled("prediction_type"),
            use_karras_sigmas=t2i.get("use_karras_sigmas", True),
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class T2IRunnerV2:
    """
    Text-in, backbone-agnostic T2I runner.

    Usage
    -----
    runner = T2IRunnerV2(T2IRunnerV2Config(backbone="sdxl"))
    img = runner.generate("a red cat beside a blue dog", cfg_scale=7.5, seed=42)
    """

    def __init__(self, config: Optional[T2IRunnerV2Config] = None) -> None:
        self.config = config or T2IRunnerV2Config()
        self._pipe = None
        self._device: Optional[str] = None
        # cached unconditional embeddings (empty string) — identical per prompt
        self._neg_cache: Optional[tuple] = None

    # -- convenience attributes for the generation manifest -------------
    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def num_inference_steps(self) -> int:
        return self.config.num_inference_steps

    @property
    def resolution(self) -> int:
        return self.config.resolution

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._pipe is not None:
            return
        cfg = self.config
        device = _resolve_device(cfg.device)
        self._device = device

        dtype = cfg.torch_dtype
        if device == "cpu" and dtype == torch.float16:
            logger.warning("float16 unsupported on CPU — using float32.")
            dtype = torch.float32

        from diffusers import DPMSolverMultistepScheduler

        if cfg.backbone == "sdxl":
            from diffusers import StableDiffusionXLPipeline
            logger.info("Loading SDXL pipeline from %s (device=%s, dtype=%s)",
                        cfg.model_id, device, dtype)
            self._pipe = StableDiffusionXLPipeline.from_pretrained(
                cfg.model_id, torch_dtype=dtype, use_safetensors=True,
                variant="fp16" if dtype == torch.float16 else None,
            ).to(device)
        elif cfg.backbone == "sd21":
            from diffusers import StableDiffusionPipeline
            logger.info("Loading SD 2.1 pipeline from %s (device=%s, dtype=%s)",
                        cfg.model_id, device, dtype)
            self._pipe = StableDiffusionPipeline.from_pretrained(
                cfg.model_id, torch_dtype=dtype,
            ).to(device)
        else:
            raise ValueError(f"Unknown backbone {cfg.backbone!r} (use 'sd21' or 'sdxl')")

        # Scheduler with explicit prediction_type (community SD2.1 repos omit it;
        # SDXL base is epsilon).
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config,
            prediction_type=cfg.prediction_type,
            use_karras_sigmas=cfg.use_karras_sigmas,
        )
        logger.info("Scheduler: DPMSolverMultistep (%s%s).",
                    cfg.prediction_type,
                    ", karras" if cfg.use_karras_sigmas else "")

        # Disable safety checker for research (sd21 only; sdxl has none).
        if hasattr(self._pipe, "safety_checker"):
            self._pipe.safety_checker = None
            self._pipe.requires_safety_checker = False

        logger.info("%s loaded on %s.", cfg.backbone, device)

    # ------------------------------------------------------------------
    # Encoding — the backbone branch
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> dict:
        """
        Encode text with the backbone's native encoder(s).

        Returns a dict of everything the pipeline call needs. Shape differs
        by backbone but callers never branch — they just splat the dict.
        """
        cfg = self.config
        with torch.no_grad():
            if cfg.backbone == "sdxl":
                (prompt_embeds, neg_embeds,
                 pooled, neg_pooled) = self._pipe.encode_prompt(
                    prompt=text,
                    device=self._device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                )
                return {
                    "prompt_embeds": prompt_embeds,
                    "negative_prompt_embeds": neg_embeds,
                    "pooled_prompt_embeds": pooled,
                    "negative_pooled_prompt_embeds": neg_pooled,
                }
            else:  # sd21
                prompt_embeds, neg_embeds = self._pipe.encode_prompt(
                    prompt=text,
                    device=self._device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=None,
                )
                return {
                    "prompt_embeds": prompt_embeds,
                    "negative_prompt_embeds": neg_embeds,
                }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_text: str,
        cfg_scale: Optional[float] = None,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate one image from raw text.

        The runner encodes internally using the backbone's native encoder(s),
        so the same call works for SD 2.1 and SDXL.
        """
        self._load()
        cfg = self.config
        scale = cfg_scale if cfg_scale is not None else cfg.default_cfg_scale
        steps = num_inference_steps or cfg.num_inference_steps

        cond = self._encode(prompt_text)

        generator = None
        if seed is not None:
            from utils.seed import get_generator
            generator = get_generator(seed, device=self._device)

        output = self._pipe(
            **cond,
            guidance_scale=scale,
            num_inference_steps=steps,
            height=cfg.resolution,
            width=cfg.resolution,
            generator=generator,
        )
        return output.images[0]

    def generate_batch(
        self,
        prompts: list[str],
        cfg_scale: Optional[float] = None,
        seeds: Optional[list[int]] = None,
        num_inference_steps: Optional[int] = None,
    ) -> list[Image.Image]:
        """Generate images for a list of text prompts (sequential, OOM-safe)."""
        seeds = seeds or [None] * len(prompts)
        return [
            self.generate(p, cfg_scale=cfg_scale, seed=s,
                          num_inference_steps=num_inference_steps)
            for p, s in zip(prompts, seeds)
        ]

    def unload(self) -> None:
        self._pipe = None
        self._neg_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("T2IRunnerV2 unloaded.")

    # ------------------------------------------------------------------
    # Denoising-trajectory instrumentation (text-in variant)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_with_trajectory(
        self,
        prompt_text: str,
        clip_scorer,
        cfg_scale: Optional[float] = None,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        score_every: int = 5,
        score_text: Optional[str] = None,
    ) -> tuple[Image.Image, list[dict]]:
        """
        Generate while scoring the partially-denoised latent every
        `score_every` steps. Works for both backbones.

        Generation is driven by `prompt_text` (the possibly-rewritten prompt).
        Scoring uses `score_text` if given, else `prompt_text`. For a fair
        cross-pipeline comparison, pass score_text = the ORIGINAL prompt so
        every pipeline is scored against the same target (user intent), not
        against its own rewrite.
        """
        self._load()
        cfg = self.config
        scale = cfg_scale if cfg_scale is not None else cfg.default_cfg_scale
        steps = num_inference_steps or cfg.num_inference_steps
        target_text = score_text if score_text is not None else prompt_text

        cond = self._encode(prompt_text)
        generator = None
        if seed is not None:
            from utils.seed import get_generator
            generator = get_generator(seed, device=self._device)

        trajectory: list[dict] = []
        pipe = self._pipe

        def _callback(pipe_ref, step: int, timestep: int, cbk: dict) -> dict:
            if step % score_every != 0 and step != steps - 1:
                return cbk
            latents = cbk["latents"]
            imgs = pipe_ref.vae.decode(
                latents / pipe_ref.vae.config.scaling_factor, return_dict=False
            )[0]
            imgs = pipe_ref.image_processor.postprocess(imgs, output_type="pil")
            try:
                s = clip_scorer.score(imgs[0], target_text)
                trajectory.append({"step": step, "clip_score": float(s)})
            except Exception as exc:
                logger.debug("Trajectory scoring failed at step %d (%s)", step, exc)
            return cbk

        output = pipe(
            **cond,
            guidance_scale=scale,
            num_inference_steps=steps,
            height=cfg.resolution,
            width=cfg.resolution,
            generator=generator,
            callback_on_step_end=_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        return output.images[0], trajectory

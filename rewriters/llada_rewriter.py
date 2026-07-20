"""
rewriters/llada_rewriter.py

Prompt rewriter using LLaDA-8B-Instruct (GSAI-ML/LLaDA-8B-Instruct).

LLaDA is a masked diffusion language model. Unlike autoregressive models it
resolves the full output token sequence jointly via iterative unmasking, making
it the principled diffusion-based counterpart to the AR baseline (Ollama/Llama).

Inference is based on the official generate() from:
  https://github.com/ML-GSAI/LLaDA/blob/main/generate.py

Key facts that shape this implementation:
  - Model loaded with AutoModel (not AutoModelForCausalLM) + trust_remote_code=True
  - Tokenizer must use LEFT padding; pad_token_id must not equal mask_id (126336)
  - Chat template applied via tokenizer.apply_chat_template for Instruct variant
  - generate() returns the full sequence (prompt + response); slice off prompt tokens
  - transformers==4.38.2 is required by the official repo
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from rewriters.base import PromptRewriter

logger = logging.getLogger(__name__)

# Token id of [MASK] in LLaDA's vocabulary — do not change.
_MASK_ID = 126336

# Expansion instruction sent to LLaDA as the user turn.
# Identical wording is used for the AR (Ollama) baseline so any difference
# in output quality is attributable to the generative mechanism, not the prompt.
_EXPANSION_INSTRUCTION = (
    "Rewrite the following image generation prompt into a richly detailed description. "
    "Explicitly name every object, assign each object its attributes (colour, size, texture, material), "
    "describe the spatial relationships between objects, and specify the overall scene composition. "
    "Output only the expanded prompt — no commentary, no explanation.\n\n"
    "Prompt: {prompt}"
)


# ---------------------------------------------------------------------------
# Official generate() — copied verbatim from ML-GSAI/LLaDA generate.py
# Only the docstring has been shortened; logic is unchanged.
# ---------------------------------------------------------------------------

def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def _generate(
    model,
    prompt: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = _MASK_ID,
) -> torch.Tensor:
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
    ).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (prompt.shape[0], gen_length),
                    dtype=attention_mask.dtype,
                    device=model.device,
                ),
            ],
            dim=-1,
        )

    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = _get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == mask_id

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                else:
                    attention_mask_ = None
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            logits_with_noise = _add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True

            x[transfer_index] = x0[transfer_index]

    return x


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLaDARewriterConfig:
    model_id: str = "GSAI-ML/LLaDA-8B-Instruct"
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.bfloat16

    # Generation hyperparameters (official defaults from generate.py)
    steps: int = 128
    gen_length: int = 128       # max tokens for the expanded prompt
    block_length: int = 32      # semi-autoregressive block size
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"

    expansion_instruction: str = _EXPANSION_INSTRUCTION
    cache_path: Optional[str] = None  # path to rewrite cache JSON file


# ---------------------------------------------------------------------------
# Rewriter
# ---------------------------------------------------------------------------

class LLaDARewriter(PromptRewriter):
    """
    Wraps LLaDA-8B-Instruct as a drop-in PromptRewriter.

    The model is loaded lazily on first call to `rewrite()` so that importing
    this module does not trigger a 16 GB weight download at startup.

    Usage
    -----
    rewriter = LLaDARewriter()
    expanded = rewriter.rewrite("a cat beside a dog")
    """

    def __init__(self, config: Optional[LLaDARewriterConfig] = None) -> None:
        self.config = config or LLaDARewriterConfig()
        self._model: Optional[AutoModel] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._cache: dict[str, str] = {}
        if self.config.cache_path:
            self._load_cache()

    def _load_cache(self) -> None:
        import json
        from pathlib import Path
        p = Path(self.config.cache_path)
        if p.exists():
            with open(p) as f:
                self._cache = json.load(f)
            logger.info("Loaded %d cached LLaDA rewrites from %s", len(self._cache), p)

    def _save_cache(self) -> None:
        import json
        from pathlib import Path
        p = Path(self.config.cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self._cache, f, indent=2)
        logger.debug("LLaDA cache saved (%d entries) → %s", len(self._cache), p)

    # ------------------------------------------------------------------
    # Lazy loader
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return

        cfg = self.config

        # Resolve device — LLaDA can technically run on CPU but 128 forward
        # passes through an 8B model takes hours. Warn loudly if CUDA is absent.
        if cfg.device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA not available — loading LLaDA on CPU. "
                "Expect extremely slow inference (hours per prompt). "
                "This is only suitable for debugging, not experiments."
            )
            cfg.device = "cpu"
            cfg.torch_dtype = torch.float32   # bfloat16 support is limited on CPU

        logger.info("Loading LLaDA tokenizer from %s", cfg.model_id)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, trust_remote_code=True
        )

        # Official requirement: left padding.
        if tokenizer.padding_side != "left":
            logger.warning(
                "Setting tokenizer.padding_side to 'left' (was '%s')",
                tokenizer.padding_side,
            )
            tokenizer.padding_side = "left"

        # Safety check from official generate.py.
        assert tokenizer.pad_token_id != _MASK_ID, (
            "pad_token_id must not equal mask_id (126336). "
            "Check the tokenizer config for this checkpoint."
        )

        logger.info("Loading LLaDA model from %s (dtype=%s)", cfg.model_id, cfg.torch_dtype)
        model = (
            AutoModel.from_pretrained(
                cfg.model_id,
                trust_remote_code=True,
                torch_dtype=cfg.torch_dtype,
                low_cpu_mem_usage=True,  # stream shards — peak host RAM ≈ 1 shard, not 16 GB
            )
            .to(cfg.device)
            .eval()
        )

        self._tokenizer = tokenizer
        self._model = model
        logger.info("LLaDA model loaded successfully.")

    # ------------------------------------------------------------------
    # PromptRewriter interface
    # ------------------------------------------------------------------

    def rewrite(self, prompt: str) -> str:
        """
        Expand a short T2I prompt into a compositionally richer description.

        Parameters
        ----------
        prompt:
            Raw user prompt, e.g. "a red cat beside a blue dog".

        Returns
        -------
        str
            Expanded prompt produced by LLaDA's masked diffusion process.
        """
        if prompt in self._cache:
            logger.debug("LLaDA cache hit for prompt: %r", prompt[:60])
            return self._cache[prompt]

        self._load()
        cfg = self.config

        user_content = cfg.expansion_instruction.format(prompt=prompt)
        messages = [{"role": "user", "content": user_content}]

        # Apply chat template (Instruct variant requires this).
        formatted = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        encoded = self._tokenizer(
            formatted,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(cfg.device)
        attention_mask = encoded["attention_mask"].to(cfg.device)

        out = _generate(
            model=self._model,
            prompt=input_ids,
            attention_mask=attention_mask,
            steps=cfg.steps,
            gen_length=cfg.gen_length,
            block_length=cfg.block_length,
            temperature=cfg.temperature,
            cfg_scale=cfg.cfg_scale,
            remasking=cfg.remasking,
        )

        # Slice off prompt tokens; decode only the generated response.
        response_ids = out[:, input_ids.shape[1]:]
        expanded = self._tokenizer.batch_decode(
            response_ids, skip_special_tokens=True
        )[0].strip()

        logger.debug("LLaDA expanded %r -> %r", prompt, expanded)
        self._cache[prompt] = expanded
        if self.config.cache_path:
            self._save_cache()
        return expanded

    def rewrite_batch(self, prompts: list[str]) -> list[str]:
        """
        Expand a batch of prompts in a single forward pass.

        Left-padding ensures correct attention masks across variable-length
        inputs, matching the official batch inference pattern.

        Parameters
        ----------
        prompts:
            List of raw user prompts.

        Returns
        -------
        list[str]
            Expanded prompts in the same order as input.
        """
        self._load()
        cfg = self.config

        formatted = [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": cfg.expansion_instruction.format(prompt=p)}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for p in prompts
        ]

        encoded = self._tokenizer(
            formatted,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(cfg.device)
        attention_mask = encoded["attention_mask"].to(cfg.device)

        out = _generate(
            model=self._model,
            prompt=input_ids,
            attention_mask=attention_mask,
            steps=cfg.steps,
            gen_length=cfg.gen_length,
            block_length=cfg.block_length,
            temperature=cfg.temperature,
            cfg_scale=cfg.cfg_scale,
            remasking=cfg.remasking,
        )

        response_ids = out[:, input_ids.shape[1]:]
        expanded = self._tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        return [e.strip() for e in expanded]

    def unload(self) -> None:
        """Release GPU memory. Call between experiment phases if needed."""
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("LLaDA model unloaded.")
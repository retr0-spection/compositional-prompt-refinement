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
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from rewriters.base import PromptRewriter
from rewriters.ollama_rewriter import RewriteTimingStats

logger = logging.getLogger(__name__)

# Token id of [MASK] in LLaDA's vocabulary — do not change.
_MASK_ID = 126336

# Expansion instruction sent to LLaDA as the user turn.
# Identical wording is used for the AR (Ollama) baseline so any difference
# in output quality is attributable to the generative mechanism, not the prompt.
#
# Design (addresses the RQ1 separation-collapse finding): earlier templates
# produced long descriptive prose that CLIP compressed into a common cluster
# (negative separation gain). This version is object-first and length-bounded:
# name every object and its binding up front (inside CLIP's effective ~20-token
# window), add only concrete visual detail after, and cap the whole thing at
# ~77 CLIP tokens so nothing important is truncated. No scene-setting,
# no mood, no narration.
_EXPANSION_INSTRUCTION = (
    "Expand this image prompt into a single concrete visual description.\n"
    "Rules:\n"
    "- Keep EVERY object from the original prompt; do not add new objects.\n"
    "- State each object with its attributes (colour, shape, material, size) "
    "and its spatial relation to the others FIRST, in the opening clause.\n"
    "- Then add only concrete, visible detail about those same objects.\n"
    "- No scene-setting, mood, lighting, camera, or narration.\n"
    "- One sentence, under 60 words.\n"
    "- Output only the description, no preamble.\n\n"
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
# Traced generate — language-layer trajectory instrumentation
# ---------------------------------------------------------------------------
# Same masked-diffusion loop as _generate, but records the partial response at
# each unmasking step so we can visualise HOW the rewrite emerges: what
# fraction of tokens are unmasked, and (optionally) how much compositional
# content the partial decode already contains. This is the language-layer
# analogue of the image denoising trajectory — and it directly visualises the
# "joint resolution" that distinguishes the diffusion mechanism from AR, since
# AR has no notion of a progressively-resolved whole sequence.

@torch.no_grad()
def _generate_traced(
    model,
    tokenizer,
    prompt: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = _MASK_ID,
    extractor=None,
) -> tuple[torch.Tensor, list[dict]]:
    """
    Like _generate, but returns (final_x, trajectory) where trajectory is a
    list of per-step dicts: {step, unmasked_fraction, [attr_count, rel_count]}.

    If `extractor` (a SemanticExtractor) is given, the partial response is
    decoded and scene-graph-counted at each step so we can show compositional
    content emerging through unmasking (slower; skip for speed).
    """
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long,
    ).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask,
             torch.ones((prompt.shape[0], gen_length),
                        dtype=attention_mask.dtype, device=model.device)],
            dim=-1,
        )

    prompt_index = x != mask_id
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    prompt_len = prompt.shape[1]

    trajectory: list[dict] = []
    global_step = 0

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length
        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = _get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                am_ = (torch.cat([attention_mask, attention_mask], dim=0)
                       if attention_mask is not None else None)
                logits = model(x_, attention_mask=am_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            logits_with_noise = _add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, -1, torch.unsqueeze(x0, -1)), -1)
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, sel = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, sel] = True
            x[transfer_index] = x0[transfer_index]

            # --- record trajectory point (row 0 only) ---
            resp = x[:, prompt_len:]
            unmasked = (resp[0] != mask_id).float().mean().item()
            rec = {"step": global_step, "unmasked_fraction": unmasked}
            if extractor is not None:
                # Decode the partial response (masked tokens -> skipped) and
                # count compositional content so far.
                partial_ids = resp[0].clone()
                partial_ids = partial_ids[partial_ids != mask_id].unsqueeze(0)
                try:
                    text = tokenizer.batch_decode(
                        partial_ids, skip_special_tokens=True)[0]
                    g = extractor.extract(text)
                    rec["attr_count"] = g["n_attribute_tokens"]
                    rec["rel_count"] = g["n_relation_tokens"]
                    rec["semantic_density"] = g["semantic_density"]
                except Exception:
                    pass
            trajectory.append(rec)
            global_step += 1

    return x, trajectory


# ---------------------------------------------------------------------------
# Product-of-Experts joint denoising (RQ5 capstone)
# ---------------------------------------------------------------------------
# Standard _generate conditions on ONE prompt. PoE conditions on MULTIPLE
# constraint prompts simultaneously and combines their per-token logits at
# every unmasking step:
#
#     log p(x) = sum_i log p_i(x)  (+ normalisation)
#
# i.e. the token distributions of N "experts" (one per constraint) are
# multiplied — a token is only confidently unmasked if ALL experts agree.
# This is the compositional operation an autoregressive model structurally
# cannot perform: AR commits tokens left-to-right conditioned on already-fixed
# outputs, so it cannot hold N constraints in superposition and resolve them
# jointly. LLaDA's parallel unmasking exposes exactly the hook PoE needs.
#
# Each expert shares the SAME masked canvas x (same tokens unmasked so far);
# they differ only in their conditioning prefix. We run one forward pass per
# expert per step, sum the log-softmax distributions, and unmask from the
# combined confidence.

@torch.no_grad()
def _generate_poe(
    model,
    expert_prompts: list[torch.Tensor],   # list of (1, Li) conditioning prefixes
    gen_length: int = 128,
    steps: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = _MASK_ID,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Joint masked-diffusion denoising under a product of N expert constraints.

    Returns the generated response ids (gen_length tokens) satisfying all
    experts jointly. Experts are conditioning prefixes (already tokenised);
    each gets its own canvas [prefix_i | shared_response], but the RESPONSE
    region is kept in lock-step across experts — the same positions unmask at
    the same step, driven by the summed (product-of-experts) confidence.
    """
    n_exp = len(expert_prompts)
    assert n_exp >= 1
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # Build one canvas per expert: [expert_prefix | masked response].
    # The response region (last gen_length tokens) is shared logically — we
    # keep it identical across experts after every unmask.
    canvases = []
    prefix_lens = []
    for pfx in expert_prompts:
        pfx = pfx.to(device)
        L = pfx.shape[1]
        prefix_lens.append(L)
        x = torch.full((1, L + gen_length), mask_id, dtype=torch.long, device=device)
        x[:, :L] = pfx.clone()
        canvases.append(x)

    # Response positions within each canvas start at prefix_len.
    def _resp_slice(e):
        return slice(prefix_lens[e], prefix_lens[e] + gen_length)

    for num_block in range(num_blocks):
        blk_lo = num_block * block_length
        blk_hi = (num_block + 1) * block_length
        # mask index over the RESPONSE region (shared shape across experts)
        resp0 = canvases[0][:, _resp_slice(0)]
        block_mask = resp0[:, blk_lo:blk_hi] == mask_id
        num_transfer = _get_num_transfer_tokens(block_mask, steps_per_block)

        for i in range(steps_per_block):
            # Response-region mask (identical across experts by construction).
            resp = canvases[0][:, _resp_slice(0)]
            mask_index = resp == mask_id  # (1, gen_length)

            # Sum log-softmax over experts = product of expert distributions.
            summed_logprobs = None
            for e, x in enumerate(canvases):
                logits = model(x).logits                    # (1, Le, vocab)
                resp_logits = logits[:, _resp_slice(e), :]  # (1, gen_length, vocab)
                lp = F.log_softmax(resp_logits.to(torch.float64), dim=-1)
                summed_logprobs = lp if summed_logprobs is None else summed_logprobs + lp

            # Renormalise the product distribution.
            combined = F.log_softmax(summed_logprobs, dim=-1)  # (1, gen_length, vocab)

            noised = _add_gumbel_noise(combined, temperature=temperature)
            x0 = torch.argmax(noised, dim=-1)                  # (1, gen_length)

            if remasking == "low_confidence":
                probs = combined.exp()
                x0_p = torch.squeeze(
                    torch.gather(probs, -1, x0.unsqueeze(-1)), -1)
            elif remasking == "random":
                x0_p = torch.rand((1, gen_length), device=device)
            else:
                raise NotImplementedError(remasking)

            # Restrict unmasking to the current block.
            x0_p[:, blk_hi:] = -np.inf
            x0_p[:, :blk_lo] = -np.inf

            keep = torch.where(mask_index, x0, resp)
            confidence = torch.where(mask_index, x0_p,
                                     torch.full_like(x0_p, -np.inf))

            transfer = torch.zeros_like(x0, dtype=torch.bool)
            k = int(num_transfer[0, i].item())
            if k > 0:
                _, sel = torch.topk(confidence[0], k=k)
                transfer[0, sel] = True

            # Apply the SAME unmask to every expert's response region so they
            # stay in lock-step (shared response, different conditioning).
            new_resp = resp.clone()
            new_resp[transfer] = keep[transfer]
            for e, x in enumerate(canvases):
                x[:, _resp_slice(e)] = new_resp

    # All experts share the response; return it from expert 0.
    return canvases[0][:, _resp_slice(0)]


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
        self.timing = RewriteTimingStats(mechanism="llada")
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

        _t0 = time.perf_counter()
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
        # Time the diffusion generate call (the real inference cost).
        # torch.cuda.synchronize ensures the timer captures GPU work, not just
        # kernel-launch time (CUDA is async).
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        self.timing.record(time.perf_counter() - _t0)

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

    def rewrite_batch(self, prompts: list[str], batch_size: int = 8) -> list[str]:
        """
        Expand a batch of prompts using chunked micro-batches.

        Why chunked: LLaDA's generate loop materialises logits of shape
        (batch, seq_len, vocab~126k) at every denoising step — at batch 500
        that is a single 60+ GB tensor (doubled again by the float64 Gumbel
        step), which segfaults the process. Micro-batches of ~8 keep the
        peak allocation in the low single-digit GBs on a 48 GB card.

        Cache behaviour: hits are returned without inference; each completed
        micro-batch is flushed to the cache file immediately, so an
        interrupted job resumes where it left off instead of starting over.

        Parameters
        ----------
        prompts:
            List of raw user prompts.
        batch_size:
            Number of prompts per forward pass. 8 is safe for gen_length=128
            on a 48 GB GPU; lower it if VRAM is tighter.

        Returns
        -------
        list[str]
            Expanded prompts in the same order as input.
        """
        cfg = self.config

        # Resolve cache hits up front; collect misses preserving order.
        results: dict[int, str] = {}
        miss_indices: list[int] = []
        for i, p in enumerate(prompts):
            if p in self._cache:
                results[i] = self._cache[p]
            else:
                miss_indices.append(i)

        if miss_indices:
            logger.info(
                "LLaDA rewrite_batch: %d cache hits, %d to generate (batch_size=%d)",
                len(prompts) - len(miss_indices), len(miss_indices), batch_size,
            )
            self._load()

        for chunk_start in range(0, len(miss_indices), batch_size):
            chunk_idx = miss_indices[chunk_start:chunk_start + batch_size]
            chunk_prompts = [prompts[i] for i in chunk_idx]

            formatted = [
                self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": cfg.expansion_instruction.format(prompt=p)}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for p in chunk_prompts
            ]

            encoded = self._tokenizer(
                formatted,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(cfg.device)
            attention_mask = encoded["attention_mask"].to(cfg.device)

            _t0 = time.perf_counter()
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
            if cfg.device == "cuda":
                torch.cuda.synchronize()
            # Record per-prompt time (chunk wall-clock / chunk size) so units
            # match the single-prompt path and AR's per-rewrite timing.
            _chunk_secs = time.perf_counter() - _t0
            for _ in chunk_idx:
                self.timing.record(_chunk_secs / len(chunk_idx))

            response_ids = out[:, input_ids.shape[1]:]
            expanded = self._tokenizer.batch_decode(response_ids, skip_special_tokens=True)

            for i, text in zip(chunk_idx, expanded):
                text = text.strip()
                results[i] = text
                self._cache[prompts[i]] = text

            # Checkpoint after every chunk — interrupted jobs resume from here.
            if cfg.cache_path:
                self._save_cache()

            done = chunk_start + len(chunk_idx)
            logger.info("LLaDA rewrite_batch: %d/%d generated.", done, len(miss_indices))

        return [results[i] for i in range(len(prompts))]

    def unload(self) -> None:
        """Release GPU memory. Call between experiment phases if needed."""
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("LLaDA model unloaded.")

    # ------------------------------------------------------------------
    # Product-of-Experts composition (RQ5 capstone)
    # ------------------------------------------------------------------

    def compose(self, constraints: list[str]) -> str:
        """
        Compose multiple constraints into ONE refined description via joint
        masked-diffusion denoising (product of experts).

        Each constraint becomes an "expert" conditioning the same shared
        response canvas; tokens are unmasked only where the experts jointly
        agree. This is the capability AR structurally lacks — it cannot hold
        N constraints in superposition and resolve them jointly.

        Parameters
        ----------
        constraints:
            List of constraint strings, e.g.
            ["a red cat", "a blue dog", "the cat is left of the dog"].

        Returns
        -------
        str
            A single composed description satisfying all constraints jointly.
        """
        self._load()
        cfg = self.config

        # Each constraint gets the expansion instruction as its expert prefix,
        # so every expert is "describe an image where <constraint>".
        expert_prefixes = []
        for c in constraints:
            msg = [{"role": "user",
                    "content": cfg.expansion_instruction.format(prompt=c)}]
            formatted = self._tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False)
            ids = self._tokenizer(formatted, add_special_tokens=False,
                                  return_tensors="pt")["input_ids"]
            expert_prefixes.append(ids)

        _t0 = time.perf_counter()
        resp_ids = _generate_poe(
            model=self._model,
            expert_prompts=expert_prefixes,
            gen_length=cfg.gen_length,
            steps=cfg.steps,
            block_length=cfg.block_length,
            temperature=cfg.temperature,
            remasking=cfg.remasking,
            device=cfg.device,
        )
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        self.timing.record(time.perf_counter() - _t0)

        composed = self._tokenizer.batch_decode(
            resp_ids, skip_special_tokens=True)[0].strip()
        logger.debug("LLaDA PoE composed %r -> %r", constraints, composed)
        return composed

    # ------------------------------------------------------------------
    # Language-layer trajectory (for the unmasking-step line graph)
    # ------------------------------------------------------------------

    def rewrite_with_trace(self, prompt: str, extractor=None) -> tuple[str, list[dict]]:
        """
        Rewrite a prompt while recording the per-unmasking-step trajectory.

        Returns (expanded_text, trajectory), where trajectory is a list of
        {step, unmasked_fraction, [attr_count, rel_count, semantic_density]}.
        Pass a SemanticExtractor to also track compositional content emerging
        through the unmasking process (slower).

        This is a diagnostic (not cached) — use on a small prompt subset to
        produce the language-layer trajectory figure.
        """
        self._load()
        cfg = self.config
        user_content = cfg.expansion_instruction.format(prompt=prompt)
        formatted = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            add_generation_prompt=True, tokenize=False)
        encoded = self._tokenizer(formatted, add_special_tokens=False,
                                  return_tensors="pt")
        input_ids = encoded["input_ids"].to(cfg.device)
        attention_mask = encoded["attention_mask"].to(cfg.device)

        out, trajectory = _generate_traced(
            model=self._model, tokenizer=self._tokenizer,
            prompt=input_ids, attention_mask=attention_mask,
            steps=cfg.steps, gen_length=cfg.gen_length,
            block_length=cfg.block_length, temperature=cfg.temperature,
            cfg_scale=cfg.cfg_scale, remasking=cfg.remasking,
            extractor=extractor,
        )
        response_ids = out[:, input_ids.shape[1]:]
        expanded = self._tokenizer.batch_decode(
            response_ids, skip_special_tokens=True)[0].strip()
        return expanded, trajectory
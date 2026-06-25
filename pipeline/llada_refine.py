"""
pipeline/llada_refine.py

LLaDAPipeline: cref = fenc(gLLaDA(t))

Rewrites the prompt using LLaDA-8B-Instruct (masked diffusion language model)
then encodes the result with the injected text encoder.

This is the proposed condition in the 2x3 experimental design. The rewriting
mechanism is a joint masked diffusion denoising process over the full token
sequence — the structural contrast to the left-to-right sequential commitment
of the AR baseline in ARPipeline.
"""

from __future__ import annotations

import logging

from encoders.base import TextEncoder
from pipeline.base import ConditioningPipeline, EncodingResult
from rewriters.llada_rewriter import LLaDARewriter, LLaDARewriterConfig

logger = logging.getLogger(__name__)


class LLaDAPipeline(ConditioningPipeline):
    """
    LLaDA diffusion-based rewrite conditioning pipeline.

    Parameters
    ----------
    encoder : TextEncoder
        A CLIPEncoder or LongCLIPEncoder instance.
    rewriter : LLaDARewriter | None
        An already-constructed LLaDARewriter, or None to use defaults.
        Pass a custom LLaDARewriterConfig to control gen_length, steps, etc.

    Example
    -------
    from encoders.clip_encoder import CLIPEncoder
    from pipeline.llada_refine import LLaDAPipeline

    pipeline = LLaDAPipeline(encoder=CLIPEncoder())
    result = pipeline.encode("a red cat beside a blue dog")

    print(result.rewritten_prompt)   # LLaDA-expanded description
    print(result.embedding.shape)    # torch.Size([1, 77, 768])
    print(result.was_truncated)      # True if expansion exceeded context window
    """

    def __init__(
        self,
        encoder: TextEncoder,
        rewriter: LLaDARewriter | None = None,
    ) -> None:
        self._encoder = encoder
        self._rewriter = rewriter or LLaDARewriter()

    @property
    def name(self) -> str:
        return f"llada_{self._encoder.name}"

    def encode(self, prompt: str) -> EncodingResult:
        rewritten = self._rewriter.rewrite(prompt)

        token_count_raw = self._encoder.token_count(prompt)
        token_count_rewritten = self._encoder.token_count(rewritten)
        context_limit = self._encoder.context_length
        was_truncated = token_count_rewritten > context_limit

        if was_truncated:
            logger.debug(
                "[%s] Rewritten prompt truncated: %d tokens > %d limit. "
                "Raw: %r | Rewritten: %r",
                self.name, token_count_rewritten, context_limit, prompt, rewritten,
            )

        embedding = self._encoder.encode(rewritten)

        return EncodingResult(
            embedding=embedding,
            raw_prompt=prompt,
            rewritten_prompt=rewritten,
            token_count_raw=token_count_raw,
            token_count_rewritten=token_count_rewritten,
            was_truncated=was_truncated,
        )

    def encode_batch(self, prompts: list[str]) -> list[EncodingResult]:
        """
        Expand and encode a batch of prompts.

        LLaDA's batch inference runs a single forward pass over all prompts
        (left-padded), so this is more efficient than calling encode() in a loop
        when processing large prompt sets.
        """
        rewritten_prompts = self._rewriter.rewrite_batch(prompts)
        results = []

        for raw, rewritten in zip(prompts, rewritten_prompts):
            token_count_raw = self._encoder.token_count(raw)
            token_count_rewritten = self._encoder.token_count(rewritten)
            context_limit = self._encoder.context_length
            was_truncated = token_count_rewritten > context_limit

            embedding = self._encoder.encode(rewritten)

            results.append(EncodingResult(
                embedding=embedding,
                raw_prompt=raw,
                rewritten_prompt=rewritten,
                token_count_raw=token_count_raw,
                token_count_rewritten=token_count_rewritten,
                was_truncated=was_truncated,
            ))

        return results

    def unload_rewriter(self) -> None:
        """
        Release LLaDA GPU memory after rewriting is complete.

        Call this before loading the T2I backbone if both models cannot
        fit in VRAM simultaneously (LLaDA is 16 GB, SD2.1 is ~5 GB).
        """
        self._rewriter.unload()
        logger.info("[%s] LLaDA rewriter unloaded.", self.name)

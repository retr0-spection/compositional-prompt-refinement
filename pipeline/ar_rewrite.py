"""
pipeline/ar_rewrite.py

ARPipeline: car = fenc(gAR(t))

Rewrites the prompt using an autoregressive LLM (Llama via Ollama) then
encodes it with the injected text encoder.

This is the mechanism-control condition in RQ4: it uses the same expansion
instruction as LLaDAPipeline so any performance difference between the two
is attributable to the generative mechanism (AR vs masked diffusion) rather
than the expansion content.
"""

from __future__ import annotations

import logging

from encoders.base import TextEncoder
from pipeline.base import ConditioningPipeline, EncodingResult
from rewriters.ollama_rewriter import OllamaRewriter, OllamaRewriterConfig

logger = logging.getLogger(__name__)


class ARPipeline(ConditioningPipeline):
    """
    Autoregressive rewrite conditioning pipeline.

    Parameters
    ----------
    encoder : TextEncoder
        A CLIPEncoder or LongCLIPEncoder instance.
    rewriter : OllamaRewriter | None
        An already-constructed OllamaRewriter, or None to use defaults.
    """

    def __init__(
        self,
        encoder: TextEncoder,
        rewriter: OllamaRewriter | None = None,
    ) -> None:
        self._encoder = encoder
        self._rewriter = rewriter or OllamaRewriter()

    @property
    def name(self) -> str:
        return f"ar_{self._encoder.name}"

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

"""
pipeline/raw.py

RawPipeline: craw = fenc(t)

Passes the user prompt directly to the text encoder with no rewriting.
This is the baseline condition in the 2x3 experimental design.
"""

from __future__ import annotations

import logging

from encoders.base import TextEncoder
from pipeline.base import ConditioningPipeline, EncodingResult

logger = logging.getLogger(__name__)


class RawPipeline(ConditioningPipeline):
    """
    Baseline conditioning pipeline — no rewriting, encode as-is.

    Parameters
    ----------
    encoder : TextEncoder
        A CLIPEncoder or LongCLIPEncoder instance.
    """

    def __init__(self, encoder: TextEncoder) -> None:
        self._encoder = encoder

    @property
    def name(self) -> str:
        return f"raw_{self._encoder.name}"

    def encode(self, prompt: str) -> EncodingResult:
        token_count = self._encoder.token_count(prompt)
        context_limit = self._encoder.context_length
        was_truncated = token_count > context_limit

        if was_truncated:
            logger.debug(
                "[%s] Prompt truncated: %d tokens > %d limit. Prompt: %r",
                self.name, token_count, context_limit, prompt,
            )

        embedding = self._encoder.encode(prompt)

        return EncodingResult(
            embedding=embedding,
            raw_prompt=prompt,
            rewritten_prompt=prompt,
            token_count_raw=token_count,
            token_count_rewritten=token_count,
            was_truncated=was_truncated,
        )

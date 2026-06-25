"""
pipeline/base.py

Abstract interface for the three conditioning strategies:
  - RawPipeline       (craw = fenc(t))
  - ARPipeline        (car  = fenc(gAR(t)))
  - LLaDAPipeline     (cref = fenc(gLLaDA(t)))

The T2I runner, evaluation modules, and experiment scripts all depend only on
this interface, so encoder and rewriter choices are fully swappable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class EncodingResult:
    """
    Everything the T2I runner and evaluation modules need from one pipeline call.

    Attributes
    ----------
    embedding : torch.Tensor
        Text conditioning tensor ready to pass to the diffusion backbone.
        Shape: (1, seq_len, hidden_dim) — matches cross-attention expectations.
    raw_prompt : str
        Original unmodified user prompt.
    rewritten_prompt : str
        Prompt after rewriting (identical to raw_prompt for RawPipeline).
    token_count_raw : int
        Token count of raw_prompt under the text encoder's tokenizer.
    token_count_rewritten : int
        Token count of rewritten_prompt. Used by truncation_monitor.
    was_truncated : bool
        True if rewritten_prompt exceeded the encoder's context window and
        was truncated before encoding.
    """

    embedding: torch.Tensor
    raw_prompt: str
    rewritten_prompt: str
    token_count_raw: int
    token_count_rewritten: int
    was_truncated: bool


class ConditioningPipeline(ABC):
    """
    Transforms a raw text prompt into a conditioning embedding.

    Subclasses wire a (optional) rewriter and a text encoder together.
    All three strategies in the 2x3 design implement this interface.
    """

    @abstractmethod
    def encode(self, prompt: str) -> EncodingResult:
        """
        Rewrite (if applicable) and encode a single prompt.

        Parameters
        ----------
        prompt : str
            Raw user prompt.

        Returns
        -------
        EncodingResult
        """

    def encode_batch(self, prompts: list[str]) -> list[EncodingResult]:
        """
        Encode a list of prompts.

        Default implementation calls encode() in a loop.
        Subclasses may override for true batch efficiency.
        """
        return [self.encode(p) for p in prompts]

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Short identifier used in logging and W&B run names.
        E.g. 'raw_clip', 'ar_longclip', 'llada_clip'.
        """
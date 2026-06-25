"""
encoders/base.py

Abstract interface for text encoders.
Both CLIPEncoder and LongCLIPEncoder implement this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class TextEncoder(ABC):
    """Encode a text string into a conditioning tensor."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'clip' or 'longclip'. Used in pipeline names."""

    @property
    @abstractmethod
    def context_length(self) -> int:
        """Maximum number of tokens this encoder accepts before truncation."""

    @abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text into a conditioning embedding.

        Parameters
        ----------
        text : str
            Prompt string (raw or rewritten).

        Returns
        -------
        torch.Tensor
            Shape: (1, seq_len, hidden_dim) — ready for cross-attention injection.
        """

    @abstractmethod
    def token_count(self, text: str) -> int:
        """
        Count tokens for text under this encoder's tokenizer.

        Used by pipelines and truncation_monitor to detect context overflow
        before (and independently of) encoding.
        """

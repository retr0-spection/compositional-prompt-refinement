"""
rewriters/base.py

Abstract interface that all prompt rewriters must implement.
Both LLaDARewriter and OllamaRewriter satisfy this contract so the
pipeline layer can swap them without any other code changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class PromptRewriter(ABC):
    """Expand or rephrase a raw prompt into a richer textual description."""

    @abstractmethod
    def rewrite(self, prompt: str) -> str:
        """
        Rewrite a single prompt.

        Parameters
        ----------
        prompt:
            Raw user prompt.

        Returns
        -------
        str
            Expanded prompt. Must be a non-empty string.
        """

    def rewrite_batch(self, prompts: list[str]) -> list[str]:
        """
        Rewrite a list of prompts.

        Default implementation calls rewrite() in a loop.
        Subclasses should override for true batch inference.
        """
        return [self.rewrite(p) for p in prompts]
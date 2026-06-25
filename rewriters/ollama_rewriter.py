"""
rewriters/ollama_rewriter.py

Prompt rewriter using an autoregressive LLM via local Ollama server.

Sends prompts to http://localhost:11434 using the Ollama chat API.
Uses the same expansion instruction as LLaDARewriter so the generative
mechanism (AR vs masked diffusion) is the only variable in RQ4.

The model is pulled automatically on first use if it is not already
downloaded — no manual `ollama pull` required.

Requirements:
    - Ollama running locally: `ollama serve`
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import requests

from rewriters.base import PromptRewriter

logger = logging.getLogger(__name__)

# Identical wording to LLaDARewriter._EXPANSION_INSTRUCTION.
# Do not change one without changing the other — they must match for RQ4.
_EXPANSION_INSTRUCTION = (
    "Rewrite the following image generation prompt into a richly detailed description. "
    "Explicitly name every object, assign each object its attributes (colour, size, texture, material), "
    "describe the spatial relationships between objects, and specify the overall scene composition. "
    "Output only the expanded prompt — no commentary, no explanation.\n\n"
    "Prompt: {prompt}"
)


@dataclass
class OllamaRewriterConfig:
    model: str = "llama3.1"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0   # deterministic; matches LLaDA temperature=0.0
    timeout: int = 600          # seconds per inference call — llama3.1 cold-start can be slow
    pull_timeout: int = 3600    # seconds for model download (llama3.1 ~4 GB)
    expansion_instruction: str = _EXPANSION_INSTRUCTION


class OllamaRewriter(PromptRewriter):
    """
    AR prompt rewriter backed by a local Ollama server.

    The configured model is pulled automatically on first use if it is not
    already present locally — behaviour mirrors how LLaDA and CLIP weights
    are fetched from HuggingFace Hub on demand.

    Uses temperature=0 by default to match the deterministic generation of
    LLaDARewriter. This ensures any quality difference in RQ4 is attributable
    to the generative mechanism rather than sampling stochasticity.

    Usage
    -----
    rewriter = OllamaRewriter()
    expanded = rewriter.rewrite("a red cat beside a blue dog")
    """

    def __init__(self, config: Optional[OllamaRewriterConfig] = None) -> None:
        self.config = config or OllamaRewriterConfig()
        self._model_ready: bool = False  # set True after first successful pull check

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def _is_model_available(self) -> bool:
        """Return True if the model is already pulled locally."""
        cfg = self.config
        try:
            resp = requests.post(
                f"{cfg.base_url}/api/show",
                json={"name": cfg.model},
                timeout=10,
            )
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def _pull_model(self) -> None:
        """
        Pull the model via the Ollama /api/pull endpoint.

        Streams progress lines so download speed is logged at INFO level.
        The pull is idempotent — Ollama skips layers it already has.
        """
        cfg = self.config
        logger.info(
            "Pulling Ollama model '%s' (this may take several minutes on first run)...",
            cfg.model,
        )
        try:
            with requests.post(
                f"{cfg.base_url}/api/pull",
                json={"name": cfg.model, "stream": True},
                stream=True,
                timeout=cfg.pull_timeout,
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        event = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    status = event.get("status", "")
                    # Log layer-download progress at DEBUG, milestones at INFO
                    if "pulling" in status and "total" in event:
                        completed = event.get("completed", 0)
                        total = event.get("total", 1)
                        pct = 100 * completed / total if total else 0
                        logger.debug("  %s — %.1f%%", status, pct)
                    elif status:
                        logger.info("  %s", status)
                    if status == "success":
                        break
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {cfg.base_url}. "
                "Run `ollama serve` before starting the pipeline."
            ) from exc
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Ollama pull failed with HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc

        logger.info("Model '%s' is ready.", cfg.model)

    def _ensure_model(self) -> None:
        """Pull the model if not already available. Called lazily on first use."""
        if self._model_ready:
            return
        if not self._is_model_available():
            self._pull_model()
        else:
            logger.info("Ollama model '%s' already present — skipping pull.", self.config.model)
        self._model_ready = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _chat(self, user_content: str) -> str:
        cfg = self.config
        payload = {
            "model": cfg.model,
            "messages": [{"role": "user", "content": user_content}],
            "stream": False,
            "options": {"temperature": cfg.temperature},
        }
        try:
            resp = requests.post(
                f"{cfg.base_url}/api/chat",
                json=payload,
                timeout=cfg.timeout,
            )
            resp.raise_for_status()
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {cfg.base_url}. "
                "Run `ollama serve` before starting the pipeline."
            ) from exc
        except requests.ReadTimeout as exc:
            raise RuntimeError(
                f"Ollama inference timed out after {cfg.timeout}s. "
                "llama3.1 cold-start (first load into VRAM) can take several minutes. "
                f"Increase timeout in OllamaRewriterConfig or experiment.yaml "
                f"(current: {cfg.timeout}s)."
            ) from exc
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc

        return resp.json()["message"]["content"].strip()

    def rewrite(self, prompt: str) -> str:
        self._ensure_model()
        user_content = self.config.expansion_instruction.format(prompt=prompt)
        expanded = self._chat(user_content)
        logger.debug("Ollama expanded %r -> %r", prompt, expanded)
        return expanded

    def rewrite_batch(self, prompts: list[str]) -> list[str]:
        """
        Expand a list of prompts sequentially.

        Ollama does not support true batch inference — each prompt is a
        separate HTTP request. For large batches, LLaDARewriter is more
        efficient as it runs a single batched forward pass.
        """
        return [self.rewrite(p) for p in prompts]

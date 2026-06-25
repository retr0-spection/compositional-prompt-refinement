"""
utils/prompt_io.py

Prompt loading, validation, and token-count logging.

Reads prompt sets from config/prompts.yaml and returns them ready for
pipeline.encode_batch(). Also provides helpers for logging expansion
ratios and token counts to W&B.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_PROMPTS_PATH = Path(__file__).parent.parent / "config" / "prompts.yaml"


def load_prompts(
    set_name: str,
    path: Optional[Path] = None,
) -> list[str]:
    """
    Load a named prompt set from the YAML config.

    Parameters
    ----------
    set_name : str
        Top-level key in prompts.yaml, e.g. 'color_binding',
        'spatial_relations', 'shape_binding', 'cats_dogs'.
    path : Path | None
        Override path to the YAML file. Defaults to config/prompts.yaml.

    Returns
    -------
    list[str]
        Flat list of prompt strings.
    """
    yaml_path = path or _DEFAULT_PROMPTS_PATH
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if set_name not in data:
        available = list(data.keys())
        raise KeyError(
            f"Prompt set {set_name!r} not found in {yaml_path}. "
            f"Available sets: {available}"
        )

    prompts = data[set_name]
    if not isinstance(prompts, list):
        raise ValueError(f"Prompt set {set_name!r} must be a YAML list, got {type(prompts)}")

    logger.info("Loaded %d prompts from set %r", len(prompts), set_name)
    return [str(p) for p in prompts]


def load_all_prompts(path: Optional[Path] = None) -> dict[str, list[str]]:
    """Load every prompt set from the YAML config."""
    yaml_path = path or _DEFAULT_PROMPTS_PATH
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return {k: [str(p) for p in v] for k, v in data.items() if isinstance(v, list)}


def log_expansion_stats(
    pipeline_name: str,
    raw_prompts: list[str],
    rewritten_prompts: list[str],
    token_counts_raw: list[int],
    token_counts_rewritten: list[int],
) -> dict[str, float]:
    """
    Compute and log prompt expansion statistics.

    Returns a flat dict suitable for wandb.log().
    """
    n = len(raw_prompts)
    assert n == len(rewritten_prompts) == len(token_counts_raw) == len(token_counts_rewritten)

    expansion_ratios = [
        t_rw / t_raw if t_raw > 0 else 1.0
        for t_raw, t_rw in zip(token_counts_raw, token_counts_rewritten)
    ]
    char_ratios = [
        len(rw) / len(raw) if len(raw) > 0 else 1.0
        for raw, rw in zip(raw_prompts, rewritten_prompts)
    ]

    stats = {
        f"{pipeline_name}/avg_token_expansion": sum(expansion_ratios) / n,
        f"{pipeline_name}/max_token_expansion": max(expansion_ratios),
        f"{pipeline_name}/avg_char_expansion": sum(char_ratios) / n,
        f"{pipeline_name}/avg_tokens_raw": sum(token_counts_raw) / n,
        f"{pipeline_name}/avg_tokens_rewritten": sum(token_counts_rewritten) / n,
    }

    logger.info(
        "[%s] Avg token expansion: %.2fx | Avg tokens raw: %.1f | rewritten: %.1f",
        pipeline_name,
        stats[f"{pipeline_name}/avg_token_expansion"],
        stats[f"{pipeline_name}/avg_tokens_raw"],
        stats[f"{pipeline_name}/avg_tokens_rewritten"],
    )
    return stats

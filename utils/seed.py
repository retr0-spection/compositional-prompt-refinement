"""utils/seed.py — Deterministic seed management."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic_cudnn: bool = True) -> None:
    """Set all RNG seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.debug("Seed set to %d", seed)


def get_generator(seed: int, device: str = "cpu") -> torch.Generator:
    """Return a seeded torch.Generator (use for diffusers' generator= arg)."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g

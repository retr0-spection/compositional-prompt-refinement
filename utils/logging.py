"""
utils/logging.py

Weights & Biases integration for experiment tracking.

All experiment scripts call init_wandb() at startup and log_metrics()
after each generation batch. The run is closed automatically via a
context manager or explicit finish().
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_run = None  # module-level W&B run handle


def init_wandb(
    project: str,
    run_name: str,
    config: dict,
    tags: Optional[list[str]] = None,
    offline: bool = False,
) -> Any:
    """
    Initialise a W&B run.

    Parameters
    ----------
    project : str
        W&B project name (from experiment.yaml: wandb_project).
    run_name : str
        Descriptive run name, e.g. 'llada_clip_rq2_seed42'.
    config : dict
        Flat dict of all hyperparameters to log.
    tags : list[str] | None
        Optional tags for filtering in the W&B dashboard.
    offline : bool
        If True, writes to local files instead of the W&B cloud.
        Useful on HPC nodes without internet access.

    Returns
    -------
    wandb.Run
    """
    global _run
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed — metrics will not be tracked remotely.")
        return None

    mode = "offline" if offline else "online"
    _run = wandb.init(
        project=project,
        name=run_name,
        config=config,
        tags=tags or [],
        mode=mode,
        reinit=True,
    )
    logger.info("W&B run initialised: %s / %s", project, run_name)
    return _run


def log_metrics(metrics: dict[str, float], step: Optional[int] = None) -> None:
    """Log a dict of scalar metrics to the current W&B run."""
    if _run is None:
        logger.debug("W&B not initialised — skipping metric log: %s", metrics)
        return
    _run.log(metrics, step=step)


def log_images(
    images: list,
    captions: Optional[list[str]] = None,
    key: str = "generated_images",
    step: Optional[int] = None,
) -> None:
    """
    Log PIL images to W&B.

    Parameters
    ----------
    images : list[PIL.Image]
    captions : list[str] | None
        One caption per image (prompt strings work well here).
    key : str
        W&B media key.
    step : int | None
    """
    if _run is None:
        return
    try:
        import wandb
    except ImportError:
        return

    wb_images = [
        wandb.Image(img, caption=cap)
        for img, cap in zip(images, captions or [""] * len(images))
    ]
    _run.log({key: wb_images}, step=step)


def finish() -> None:
    """Close the current W&B run."""
    global _run
    if _run is not None:
        _run.finish()
        _run = None

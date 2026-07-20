"""
utils/naming.py

Canonical naming for generated images and per-directory generation manifests.

Encoding generation settings into filenames makes the disk cache
self-invalidating: if cfg_scale or seed changes, the resume logic finds no
matching file and regenerates, instead of silently reusing images produced
under different settings.

Run-level settings that would bloat filenames (model id, sampler steps,
resolution) go into a generation_meta.json manifest per image directory;
a mismatch between the manifest and current settings logs a loud warning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

META_FILENAME = "generation_meta.json"


def img_name(idx: int, cfg_scale: float, seed: int) -> str:
    """
    Canonical image filename embedding per-image generation variables.

    Example: prompt_007_cfg7.5_seed42.png
    ``%g`` formatting keeps floats stable (7.5 -> '7.5', 7.0 -> '7').
    """
    return f"prompt_{idx:03d}_cfg{cfg_scale:g}_seed{seed}.png"


def write_meta(img_dir: Path, settings: dict) -> None:
    """Write the run-level generation manifest for an image directory."""
    path = Path(img_dir) / META_FILENAME
    with open(path, "w") as f:
        json.dump(settings, f, indent=2, default=str, sort_keys=True)
    logger.debug("Generation manifest written to %s", path)


def check_meta(img_dir: Path, settings: dict) -> bool:
    """
    Compare current run-level settings against an existing manifest.

    Returns True if compatible (no manifest yet, or all shared keys match).
    Returns False and logs a warning listing mismatched keys otherwise —
    caller decides whether to proceed, but cached images in a mismatched
    directory should be treated as stale.
    """
    path = Path(img_dir) / META_FILENAME
    if not path.exists():
        return True
    try:
        with open(path) as f:
            existing = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Unreadable manifest %s (%s) — treating as compatible.", path, exc)
        return True

    mismatches = {
        k: (existing[k], settings[k])
        for k in settings.keys() & existing.keys()
        if str(existing[k]) != str(settings[k])
    }
    if mismatches:
        logger.warning(
            "Generation settings mismatch in %s — cached images may be STALE. "
            "Mismatched keys: %s. Delete the directory to force regeneration.",
            img_dir, mismatches,
        )
        return False
    return True

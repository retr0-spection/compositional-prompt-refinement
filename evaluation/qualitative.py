"""
evaluation/qualitative.py

Qualitative evaluation support: rater annotation loading and Cohen's κ.

Raters annotate generated images on a structured rubric. This module
loads their responses (from CSV), computes inter-rater agreement (Cohen's κ),
and aggregates per-pipeline scores.

Expected CSV schema
-------------------
image_id, pipeline_name, rater_id, attribute_correct (0/1),
relation_correct (0/1), overall_quality (1-5), notes
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RaterAnnotation:
    image_id: str
    pipeline_name: str
    rater_id: str
    attribute_correct: int       # 0 or 1
    relation_correct: int        # 0 or 1
    overall_quality: int         # 1–5
    notes: str = ""


def load_annotations(csv_path: str | Path) -> list[RaterAnnotation]:
    """
    Load rater annotations from a CSV file.

    Parameters
    ----------
    csv_path : str | Path
        Path to the annotations CSV.

    Returns
    -------
    list[RaterAnnotation]
    """
    annotations = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations.append(RaterAnnotation(
                image_id=row["image_id"].strip(),
                pipeline_name=row["pipeline_name"].strip(),
                rater_id=row["rater_id"].strip(),
                attribute_correct=int(row["attribute_correct"]),
                relation_correct=int(row["relation_correct"]),
                overall_quality=int(row["overall_quality"]),
                notes=row.get("notes", "").strip(),
            ))
    logger.info("Loaded %d annotations from %s", len(annotations), csv_path)
    return annotations


def cohens_kappa(rater_a: list[int], rater_b: list[int]) -> float:
    """
    Compute Cohen's κ for two raters over binary or ordinal labels.

    Parameters
    ----------
    rater_a, rater_b : list[int]
        Paired ratings for the same set of items. Must be the same length.

    Returns
    -------
    float
        κ ∈ [-1, 1]. Values ≥ 0.6 are typically considered substantial agreement.
    """
    assert len(rater_a) == len(rater_b), "Rater lists must be the same length"
    n = len(rater_a)
    if n == 0:
        return 0.0

    labels = sorted(set(rater_a) | set(rater_b))
    k = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    # Confusion matrix
    conf = np.zeros((k, k), dtype=float)
    for a, b in zip(rater_a, rater_b):
        conf[label_to_idx[a], label_to_idx[b]] += 1

    p_o = np.trace(conf) / n  # observed agreement
    row_sums = conf.sum(axis=1) / n
    col_sums = conf.sum(axis=0) / n
    p_e = np.dot(row_sums, col_sums)  # expected agreement by chance

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def compute_inter_rater_agreement(
    annotations: list[RaterAnnotation],
    field: str = "attribute_correct",
) -> dict[str, float]:
    """
    Compute pairwise Cohen's κ for all rater pairs.

    Groups annotations by image_id, then computes κ for each pair of raters
    that annotated the same images.

    Parameters
    ----------
    annotations : list[RaterAnnotation]
    field : str
        Which field to compute agreement on: 'attribute_correct',
        'relation_correct', or 'overall_quality'.

    Returns
    -------
    dict mapping rater_pair_key → κ, plus an 'average_kappa' entry.
    """
    # Group by image_id → rater_id → score
    by_image: dict[str, dict[str, int]] = defaultdict(dict)
    for ann in annotations:
        val = getattr(ann, field)
        by_image[ann.image_id][ann.rater_id] = val

    # Find all rater pairs
    all_raters = sorted(set(ann.rater_id for ann in annotations))
    pair_kappas = {}

    for i in range(len(all_raters)):
        for j in range(i + 1, len(all_raters)):
            r_a, r_b = all_raters[i], all_raters[j]
            shared_images = [
                img_id for img_id, raters in by_image.items()
                if r_a in raters and r_b in raters
            ]
            if not shared_images:
                continue
            scores_a = [by_image[img_id][r_a] for img_id in shared_images]
            scores_b = [by_image[img_id][r_b] for img_id in shared_images]
            kappa = cohens_kappa(scores_a, scores_b)
            key = f"kappa_{r_a}_vs_{r_b}"
            pair_kappas[key] = kappa
            logger.info("κ(%s vs %s) on '%s': %.3f (n=%d)", r_a, r_b, field, kappa, len(shared_images))

    if pair_kappas:
        pair_kappas["average_kappa"] = float(np.mean(list(pair_kappas.values())))
    return pair_kappas


def aggregate_by_pipeline(
    annotations: list[RaterAnnotation],
) -> dict[str, dict[str, float]]:
    """
    Compute per-pipeline mean scores across all raters.

    Returns
    -------
    dict mapping pipeline_name → {avg_attr_correct, avg_rel_correct, avg_quality}
    """
    by_pipeline: dict[str, list[RaterAnnotation]] = defaultdict(list)
    for ann in annotations:
        by_pipeline[ann.pipeline_name].append(ann)

    out = {}
    for pipeline, anns in sorted(by_pipeline.items()):
        out[pipeline] = {
            "avg_attribute_correct": np.mean([a.attribute_correct for a in anns]),
            "avg_relation_correct": np.mean([a.relation_correct for a in anns]),
            "avg_overall_quality": np.mean([a.overall_quality for a in anns]),
            "n_annotations": len(anns),
        }
        logger.info(
            "[%s] Attr=%.3f | Rel=%.3f | Quality=%.2f (n=%d)",
            pipeline,
            out[pipeline]["avg_attribute_correct"],
            out[pipeline]["avg_relation_correct"],
            out[pipeline]["avg_overall_quality"],
            out[pipeline]["n_annotations"],
        )
    return out

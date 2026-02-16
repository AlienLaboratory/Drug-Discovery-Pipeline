"""Similarity-based virtual screening.

Screens a library against a set of reference actives using
fingerprint similarity. Wraps the Phase 1 similarity module.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from drugflow.core.constants import (
    DEFAULT_SIMILARITY_FP_TYPE,
    DEFAULT_SIMILARITY_METRIC,
    DEFAULT_SIMILARITY_THRESHOLD,
)
from drugflow.core.exceptions import ScreeningError
from drugflow.core.logging import get_logger, progress_bar
from drugflow.core.models import MoleculeDataset, MoleculeRecord
from drugflow.phase1.analysis.similarity import get_similarity_func

logger = get_logger("screening.similarity")


def compute_max_similarity(
    rec: MoleculeRecord,
    reference_fps: List[np.ndarray],
    fp_type: str = DEFAULT_SIMILARITY_FP_TYPE,
    metric: str = DEFAULT_SIMILARITY_METRIC,
) -> float:
    """Compute maximum similarity of a molecule against reference fingerprints.

    Parameters
    ----------
    rec : MoleculeRecord
        Query molecule record (must have fingerprints computed).
    reference_fps : list of np.ndarray
        Reference active fingerprints.
    fp_type : str
        Fingerprint key in rec.fingerprints.
    metric : str
        Similarity metric name.

    Returns
    -------
    float
        Maximum similarity to any reference molecule.
    """
    if fp_type not in rec.fingerprints:
        return 0.0

    sim_func = get_similarity_func(metric)
    query_fp = rec.fingerprints[fp_type]

    max_sim = 0.0
    for ref_fp in reference_fps:
        sim = sim_func(query_fp, ref_fp)
        if sim > max_sim:
            max_sim = sim
    return max_sim


def compute_mean_similarity(
    rec: MoleculeRecord,
    reference_fps: List[np.ndarray],
    fp_type: str = DEFAULT_SIMILARITY_FP_TYPE,
    metric: str = DEFAULT_SIMILARITY_METRIC,
) -> float:
    """Compute mean similarity of a molecule against reference fingerprints.

    Parameters
    ----------
    rec : MoleculeRecord
        Query molecule record.
    reference_fps : list of np.ndarray
        Reference active fingerprints.
    fp_type : str
        Fingerprint key.
    metric : str
        Similarity metric name.

    Returns
    -------
    float
        Mean similarity to all reference molecules.
    """
    if fp_type not in rec.fingerprints or not reference_fps:
        return 0.0

    sim_func = get_similarity_func(metric)
    query_fp = rec.fingerprints[fp_type]

    sims = [sim_func(query_fp, ref_fp) for ref_fp in reference_fps]
    return float(np.mean(sims))


def extract_reference_fps(
    reference_dataset: MoleculeDataset,
    fp_type: str = DEFAULT_SIMILARITY_FP_TYPE,
) -> List[np.ndarray]:
    """Extract fingerprint arrays from a reference dataset.

    Parameters
    ----------
    reference_dataset : MoleculeDataset
        Dataset of reference active molecules.
    fp_type : str
        Fingerprint key to extract.

    Returns
    -------
    list of np.ndarray
        List of fingerprint arrays.

    Raises
    ------
    ScreeningError
        If no fingerprints are found.
    """
    fps = []
    for rec in reference_dataset.valid_records:
        if fp_type in rec.fingerprints:
            fps.append(rec.fingerprints[fp_type])

    if not fps:
        raise ScreeningError(
            f"No fingerprints of type '{fp_type}' found in reference dataset. "
            f"Compute fingerprints first."
        )
    return fps


def screen_similarity(
    library: MoleculeDataset,
    reference: MoleculeDataset,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    fp_type: str = DEFAULT_SIMILARITY_FP_TYPE,
    metric: str = DEFAULT_SIMILARITY_METRIC,
    aggregation: str = "max",
) -> MoleculeDataset:
    """Screen library molecules by similarity to reference actives.

    Stores results in rec.properties:
      - "sim_screen_max": maximum similarity to any reference
      - "sim_screen_mean": mean similarity to all references
      - "sim_screen_pass": True if above threshold

    Parameters
    ----------
    library : MoleculeDataset
        Library to screen.
    reference : MoleculeDataset
        Reference actives dataset (must have fingerprints).
    threshold : float
        Minimum similarity threshold (0-1).
    fp_type : str
        Fingerprint type to use.
    metric : str
        Similarity metric.
    aggregation : str
        "max" uses maximum similarity, "mean" uses mean similarity.

    Returns
    -------
    MoleculeDataset
        New dataset with only hits above threshold.

    Raises
    ------
    ScreeningError
        If reference has no fingerprints or invalid parameters.
    """
    if threshold < 0 or threshold > 1:
        raise ScreeningError(f"Threshold must be 0-1, got {threshold}")
    if aggregation not in ("max", "mean"):
        raise ScreeningError(f"aggregation must be 'max' or 'mean', got '{aggregation}'")

    reference_fps = extract_reference_fps(reference, fp_type)
    logger.info(
        f"Similarity screening: {len(library.valid_records)} library mols "
        f"vs {len(reference_fps)} reference actives "
        f"(threshold={threshold}, metric={metric}, agg={aggregation})"
    )

    hits = []

    for rec in progress_bar(library.valid_records, desc="Similarity screening"):
        if rec.mol is None or fp_type not in rec.fingerprints:
            continue

        max_sim = compute_max_similarity(rec, reference_fps, fp_type, metric)
        mean_sim = compute_mean_similarity(rec, reference_fps, fp_type, metric)

        rec.properties["sim_screen_max"] = max_sim
        rec.properties["sim_screen_mean"] = mean_sim

        # Apply threshold based on aggregation
        if aggregation == "max":
            passes = max_sim >= threshold
        else:
            passes = mean_sim >= threshold

        rec.properties["sim_screen_pass"] = passes
        rec.add_provenance(
            f"screen:similarity:{'hit' if passes else 'miss'}"
        )

        if passes:
            hits.append(rec)

    logger.info(
        f"Similarity screen: {len(hits)} hits from "
        f"{len(library.valid_records)} molecules"
    )

    result = MoleculeDataset(records=hits, name=f"{library.name}_sim_hits")
    result._provenance = library._provenance + [
        f"screen:similarity:threshold={threshold}"
    ]
    return result

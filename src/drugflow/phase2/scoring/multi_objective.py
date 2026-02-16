"""Multi-objective weighted scoring.

Combines predicted activity, drug-likeness, and synthetic accessibility
into a single composite score for ranking molecules.
"""

from typing import Dict, List, Optional

import numpy as np

from drugflow.core.constants import SCORING_DEFAULT_WEIGHTS
from drugflow.core.exceptions import ScoringError
from drugflow.core.logging import get_logger, progress_bar
from drugflow.core.models import MoleculeDataset, MoleculeRecord

logger = get_logger("scoring.multi_objective")


def compute_composite_score(
    rec: MoleculeRecord,
    weights: Optional[Dict[str, float]] = None,
    higher_is_better: Optional[Dict[str, bool]] = None,
) -> float:
    """Compute weighted composite score for a molecule.

    Parameters
    ----------
    rec : MoleculeRecord
        Molecule with all component scores in properties.
    weights : dict, optional
        Mapping of score component to weight. Components:
        - "predicted_activity" -> rec.properties["predicted_activity"]
        - "drug_likeness" -> rec.properties["drug_likeness_score"]
        - "sa_score" -> rec.properties["sa_score_normalized"]
    higher_is_better : dict, optional
        Whether higher values are better for each component.
        Defaults: all True (after normalization).

    Returns
    -------
    float
        Composite score (0-1 scale).
    """
    if weights is None:
        weights = SCORING_DEFAULT_WEIGHTS.copy()

    if higher_is_better is None:
        higher_is_better = {
            "predicted_activity": True,
            "drug_likeness": True,
            "sa_score": True,  # already normalized (1=easy)
        }

    # Property key mapping
    property_keys = {
        "predicted_activity": "predicted_activity",
        "drug_likeness": "drug_likeness_score",
        "sa_score": "sa_score_normalized",
    }

    score = 0.0
    total_weight = 0.0

    for component, weight in weights.items():
        prop_key = property_keys.get(component, component)
        value = rec.properties.get(prop_key)

        if value is None:
            continue

        value = float(value)

        # Ensure value is in 0-1 range for consistent combination
        # (activity may not be pre-normalized)
        if not higher_is_better.get(component, True):
            value = 1.0 - value

        score += value * weight
        total_weight += weight

    if total_weight > 0:
        score /= total_weight

    return float(score)


def normalize_scores_minmax(
    dataset: MoleculeDataset,
    property_name: str,
) -> None:
    """Normalize a property across the dataset to 0-1 range using min-max.

    Modifies records in-place, adding "{property_name}_normalized".

    Parameters
    ----------
    dataset : MoleculeDataset
        Input dataset.
    property_name : str
        Property key to normalize.
    """
    values = []
    for rec in dataset.valid_records:
        val = rec.properties.get(property_name)
        if val is not None:
            values.append(float(val))

    if not values:
        return

    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val

    for rec in dataset.valid_records:
        val = rec.properties.get(property_name)
        if val is not None:
            if range_val > 0:
                normalized = (float(val) - min_val) / range_val
            else:
                normalized = 0.5
            rec.properties[f"{property_name}_normalized"] = normalized


def compute_composite_score_dataset(
    dataset: MoleculeDataset,
    weights: Optional[Dict[str, float]] = None,
    normalize_activity: bool = True,
) -> MoleculeDataset:
    """Compute composite scores for all valid records.

    Stores result in rec.properties["composite_score"].

    Parameters
    ----------
    dataset : MoleculeDataset
        Dataset with predicted_activity, drug_likeness_score,
        and sa_score_normalized in properties.
    weights : dict, optional
        Component weights.
    normalize_activity : bool
        If True, normalize predicted_activity to 0-1 before combining.

    Returns
    -------
    MoleculeDataset
        Dataset with composite scores added.
    """
    # Normalize activity predictions if needed
    if normalize_activity:
        normalize_scores_minmax(dataset, "predicted_activity")

    # Build effective weights — use normalized activity key if available
    effective_weights = (weights or SCORING_DEFAULT_WEIGHTS).copy()

    count = 0
    for rec in progress_bar(dataset.valid_records, desc="Computing composite scores"):
        try:
            # Use normalized activity if available
            if normalize_activity and "predicted_activity_normalized" in rec.properties:
                # Temporarily swap for composite calculation
                original = rec.properties.get("predicted_activity")
                rec.properties["predicted_activity"] = rec.properties[
                    "predicted_activity_normalized"
                ]
                score = compute_composite_score(rec, effective_weights)
                # Restore original
                if original is not None:
                    rec.properties["predicted_activity"] = original
            else:
                score = compute_composite_score(rec, effective_weights)

            rec.properties["composite_score"] = score
            rec.add_provenance("composite_score:computed")
            count += 1
        except Exception as e:
            logger.warning(
                f"Composite score failed for {rec.record_id}: {e}"
            )

    logger.info(f"Computed composite scores for {count} molecules")
    return dataset

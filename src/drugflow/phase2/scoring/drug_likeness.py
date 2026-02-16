"""Aggregate drug-likeness scoring.

Combines QED (Quantitative Estimate of Drug-likeness) with
filter pass/fail results into a single drug-likeness score.
"""

from typing import Dict, List, Optional

from rdkit import Chem
from rdkit.Chem import QED as QED_module

from drugflow.core.exceptions import ScoringError
from drugflow.core.logging import get_logger, progress_bar
from drugflow.core.models import MoleculeDataset, MoleculeRecord

logger = get_logger("scoring.drug_likeness")


def compute_drug_likeness(
    rec: MoleculeRecord,
    use_qed: bool = True,
    filter_weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute aggregate drug-likeness score for a molecule.

    Combines QED with filter results into a 0-1 score where
    1 = most drug-like.

    Parameters
    ----------
    rec : MoleculeRecord
        Molecule record (should have properties and optionally filter results).
    use_qed : bool
        Whether to include QED as a component.
    filter_weights : dict, optional
        Weights for filter pass/fail contributions.
        Defaults to {"lipinski": 0.3, "pains": 0.2, "veber": 0.1}.

    Returns
    -------
    float
        Drug-likeness score (0-1).
    """
    if filter_weights is None:
        filter_weights = {
            "lipinski": 0.3,
            "pains": 0.2,
            "veber": 0.1,
        }

    score = 0.0
    total_weight = 0.0

    # QED component
    if use_qed:
        qed_val = rec.properties.get("QED")
        if qed_val is None and rec.mol is not None:
            qed_val = QED_module.qed(rec.mol)
            rec.properties["QED"] = qed_val

        if qed_val is not None:
            qed_weight = 0.4
            score += qed_val * qed_weight
            total_weight += qed_weight

    # Filter components
    for filter_name, weight in filter_weights.items():
        key = f"{filter_name}_pass"
        if key in rec.properties:
            filter_score = 1.0 if rec.properties[key] else 0.0
            score += filter_score * weight
            total_weight += weight

    if total_weight > 0:
        score /= total_weight

    return float(score)


def compute_drug_likeness_dataset(
    dataset: MoleculeDataset,
    use_qed: bool = True,
    filter_weights: Optional[Dict[str, float]] = None,
) -> MoleculeDataset:
    """Compute drug-likeness score for all valid records.

    Stores result in rec.properties["drug_likeness_score"].

    Parameters
    ----------
    dataset : MoleculeDataset
        Input dataset (ideally with filters already applied).
    use_qed : bool
        Whether to include QED.
    filter_weights : dict, optional
        Filter contribution weights.

    Returns
    -------
    MoleculeDataset
        Dataset with drug-likeness scores added.
    """
    count = 0
    for rec in progress_bar(dataset.valid_records, desc="Computing drug-likeness"):
        if rec.mol is None:
            continue
        try:
            score = compute_drug_likeness(rec, use_qed, filter_weights)
            rec.properties["drug_likeness_score"] = score
            rec.add_provenance("drug_likeness:computed")
            count += 1
        except Exception as e:
            logger.warning(
                f"Drug-likeness failed for {rec.record_id}: {e}"
            )

    logger.info(f"Computed drug-likeness for {count} molecules")
    return dataset

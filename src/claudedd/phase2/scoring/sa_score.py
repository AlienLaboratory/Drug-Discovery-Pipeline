"""Synthetic Accessibility Score computation.

Uses the Ertl & Schuffenhauer SA Score from RDKit's Contrib module.
Score ranges from 1 (easy to synthesize) to 10 (hard to synthesize).
"""

from typing import Optional

from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer

from claudedd.core.constants import SA_SCORE_MAX, SA_SCORE_MIN
from claudedd.core.exceptions import ScoringError
from claudedd.core.logging import get_logger, progress_bar
from claudedd.core.models import MoleculeDataset, MoleculeRecord

logger = get_logger("scoring.sa_score")


def compute_sa_score(mol: Chem.Mol) -> float:
    """Compute synthetic accessibility score for a single molecule.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object.

    Returns
    -------
    float
        SA score between 1 (easy) and 10 (hard).

    Raises
    ------
    ScoringError
        If computation fails.
    """
    if mol is None:
        raise ScoringError("Cannot compute SA score for None molecule")
    try:
        score = sascorer.calculateScore(mol)
        return float(score)
    except Exception as e:
        raise ScoringError(f"SA score computation failed: {e}")


def normalize_sa_score(sa_score: float) -> float:
    """Normalize SA score to 0-1 range where 1 = easy to synthesize.

    Parameters
    ----------
    sa_score : float
        Raw SA score (1-10).

    Returns
    -------
    float
        Normalized score (0-1, higher = more synthesizable).
    """
    # Invert: 1 (easy) -> 1.0, 10 (hard) -> 0.0
    return max(0.0, min(1.0,
        (SA_SCORE_MAX - sa_score) / (SA_SCORE_MAX - SA_SCORE_MIN)
    ))


def compute_sa_score_dataset(
    dataset: MoleculeDataset,
    normalize: bool = True,
) -> MoleculeDataset:
    """Compute SA score for all valid records in a dataset.

    Stores results in rec.properties:
      - "sa_score": raw SA score (1-10)
      - "sa_score_normalized": normalized score (0-1, if normalize=True)

    Parameters
    ----------
    dataset : MoleculeDataset
        Input dataset with valid molecules.
    normalize : bool
        Whether to also compute the normalized (0-1) score.

    Returns
    -------
    MoleculeDataset
        Same dataset with SA scores added to properties.
    """
    count = 0
    for rec in progress_bar(dataset.valid_records, desc="Computing SA scores"):
        if rec.mol is None:
            continue
        try:
            score = compute_sa_score(rec.mol)
            rec.properties["sa_score"] = score
            if normalize:
                rec.properties["sa_score_normalized"] = normalize_sa_score(score)
            rec.add_provenance("sa_score:computed")
            count += 1
        except ScoringError as e:
            logger.warning(f"SA score failed for {rec.record_id}: {e}")

    logger.info(f"Computed SA scores for {count} molecules")
    return dataset

"""ADMET prediction pipeline: batch prediction for molecule datasets.

Orchestrates all ADMET domain modules (absorption, distribution,
metabolism, excretion, toxicity) and scoring into a single pipeline.
"""

from typing import Dict, Optional

from rdkit import Chem

from drugflow.core.exceptions import ADMETError
from drugflow.core.logging import get_logger, progress_bar
from drugflow.core.models import MoleculeDataset, MoleculeRecord
from drugflow.phase5.admet.absorption import predict_absorption, predict_absorption_record
from drugflow.phase5.admet.distribution import predict_distribution_record
from drugflow.phase5.admet.excretion import predict_excretion_record
from drugflow.phase5.admet.metabolism import predict_metabolism_record
from drugflow.phase5.admet.toxicity import predict_toxicity_record
from drugflow.phase5.admet.scoring import compute_admet_score_record

logger = get_logger("admet.pipeline")


def predict_admet(
    mol: Chem.Mol,
    weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """Predict all ADMET properties for a single molecule.

    Runs absorption, distribution, metabolism, excretion, and toxicity
    predictions, then computes aggregate ADMET score.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.
    weights : dict, optional
        Custom weights for ADMET score components.

    Returns
    -------
    dict
        All ADMET predictions as a flat dict with admet_ prefix keys.

    Raises
    ------
    ADMETError
        If molecule is None or prediction fails.
    """
    if mol is None:
        raise ADMETError("Cannot predict ADMET for None molecule")

    # Create a temporary record to use the record-based API
    rec = MoleculeRecord(mol=mol)

    try:
        predict_absorption_record(rec)
        predict_distribution_record(rec)
        predict_metabolism_record(rec)
        predict_excretion_record(rec)
        predict_toxicity_record(rec)
        compute_admet_score_record(rec, weights)
    except Exception as e:
        raise ADMETError(f"ADMET prediction failed: {e}") from e

    # Return only admet_ prefixed properties
    return {k: v for k, v in rec.properties.items() if k.startswith("admet_")}


def predict_admet_dataset(
    dataset: MoleculeDataset,
    weights: Optional[Dict[str, float]] = None,
) -> MoleculeDataset:
    """Run ADMET predictions for all valid records in a dataset.

    Stores results in rec.properties with admet_ prefix.
    Adds provenance "admet:predicted".

    Parameters
    ----------
    dataset : MoleculeDataset
        Input dataset with valid molecules.
    weights : dict, optional
        Custom ADMET scoring weights.

    Returns
    -------
    MoleculeDataset
        Dataset with ADMET properties added to each record.
    """
    count = 0
    failed = 0
    for rec in progress_bar(dataset.valid_records, desc="Predicting ADMET"):
        if rec.mol is None:
            continue
        try:
            predict_absorption_record(rec)
            predict_distribution_record(rec)
            predict_metabolism_record(rec)
            # Excretion depends on metabolism (metabolic_stability_score)
            predict_excretion_record(rec)
            predict_toxicity_record(rec)
            compute_admet_score_record(rec, weights)
            rec.add_provenance("admet:predicted")
            count += 1
        except Exception as e:
            logger.warning(f"ADMET prediction failed for {rec.record_id}: {e}")
            failed += 1

    logger.info(f"Predicted ADMET for {count} molecules ({failed} failed)")
    return dataset

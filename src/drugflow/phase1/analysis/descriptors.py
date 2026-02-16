"""Molecular descriptor computation using RDKit.

Computes the full set of ~208 RDKit 2D descriptors or a user-specified subset.
"""

from typing import Dict, List, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from drugflow.core.constants import DEFAULT_DESCRIPTORS
from drugflow.core.logging import get_logger, progress_bar
from drugflow.core.models import MoleculeDataset

logger = get_logger("analysis.descriptors")


def compute_descriptors_dataset(
    dataset: MoleculeDataset,
    descriptor_names: Optional[List[str]] = None,
    compute_all: bool = False,
) -> MoleculeDataset:
    """Compute molecular descriptors for all valid records."""
    if compute_all:
        names = get_available_descriptors()
    elif descriptor_names:
        names = descriptor_names
    else:
        names = DEFAULT_DESCRIPTORS

    count = 0
    for rec in progress_bar(dataset.valid_records, desc="Computing descriptors"):
        if rec.mol is None:
            continue
        try:
            desc = compute_descriptors_single(rec.mol, descriptor_names=names)
            rec.descriptors.update(desc)
            rec.properties.update(desc)
            rec.add_provenance("descriptors:computed")
            count += 1
        except Exception as e:
            logger.warning(
                f"Descriptor computation failed for {rec.record_id}: {e}"
            )

    logger.info(f"Computed {len(names)} descriptors for {count} molecules")
    return dataset


def compute_descriptors_single(
    mol: Chem.Mol,
    descriptor_names: Optional[List[str]] = None,
    compute_all: bool = False,
) -> Dict[str, float]:
    """Compute descriptors for a single RDKit Mol object."""
    if compute_all:
        descriptor_names = get_available_descriptors()
    elif descriptor_names is None:
        descriptor_names = DEFAULT_DESCRIPTORS

    # Build a calculator for the requested descriptors
    available = {name for name, _ in Descriptors._descList}
    valid_names = [n for n in descriptor_names if n in available]

    if not valid_names:
        return {}

    calc = MoleculeDescriptors.MolecularDescriptorCalculator(valid_names)
    values = calc.CalcDescriptors(mol)

    result = {}
    for name, val in zip(valid_names, values):
        result[name] = float(val) if val is not None else float("nan")

    return result


def get_available_descriptors() -> List[str]:
    """Return list of all available RDKit descriptor names."""
    return [name for name, _ in Descriptors._descList]

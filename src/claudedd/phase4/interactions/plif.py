"""Protein-Ligand Interaction Fingerprints (PLIF).

Encodes protein-ligand interactions as binary fingerprints for
comparison and clustering of binding modes.
"""

import logging
from typing import Optional

import numpy as np
from rdkit import Chem

from claudedd.core.exceptions import DockingError
from claudedd.core.models import MoleculeDataset
from claudedd.phase4.interactions.contacts import detect_all_contacts

logger = logging.getLogger(__name__)


def compute_plif(
    ligand: Chem.Mol,
    protein: Chem.Mol,
    conf_id_lig: int = 0,
    conf_id_prot: int = 0,
) -> np.ndarray:
    """Compute protein-ligand interaction fingerprint.

    Creates a binary fingerprint encoding which interaction types
    are present. Format: [n_hbonds_donor, n_hbonds_acceptor,
    n_hydrophobic, n_pi_stacking, n_salt_bridges, has_hbond,
    has_hydrophobic, has_pi, has_salt_bridge].

    Args:
        ligand: Ligand with 3D coords.
        protein: Protein with 3D coords.
        conf_id_lig: Ligand conformer ID.
        conf_id_prot: Protein conformer ID.

    Returns:
        Binary/count interaction fingerprint as numpy array.
    """
    contacts = detect_all_contacts(ligand, protein, conf_id_lig, conf_id_prot)

    # Count interactions by type
    n_hbond_donor = sum(
        1 for hb in contacts["hbonds"] if hb["type"] == "ligand_donor"
    )
    n_hbond_acceptor = sum(
        1 for hb in contacts["hbonds"] if hb["type"] == "ligand_acceptor"
    )
    n_hydrophobic = len(contacts["hydrophobic"])
    n_pi_stacking = len(contacts["pi_stacking"])
    n_salt_bridges = len(contacts["salt_bridges"])

    plif = np.array([
        n_hbond_donor,
        n_hbond_acceptor,
        n_hydrophobic,
        n_pi_stacking,
        n_salt_bridges,
        float(n_hbond_donor > 0 or n_hbond_acceptor > 0),  # has any hbond
        float(n_hydrophobic > 0),
        float(n_pi_stacking > 0),
        float(n_salt_bridges > 0),
    ])

    return plif


def compute_plif_dataset(
    dataset: MoleculeDataset,
    protein: Chem.Mol,
    conf_id_prot: int = 0,
) -> MoleculeDataset:
    """Compute PLIF for all molecules in a dataset.

    Args:
        dataset: Dataset with 3D molecules.
        protein: Protein molecule.
        conf_id_prot: Protein conformer ID.

    Returns:
        Dataset with PLIF stored in properties.
    """
    for rec in dataset.valid_records:
        if rec.mol is None or rec.mol.GetNumConformers() == 0:
            continue
        try:
            plif = compute_plif(rec.mol, protein, conf_id_prot=conf_id_prot)
            rec.properties["plif"] = plif.tolist()
            rec.properties["hbond_count"] = int(plif[0] + plif[1])
            rec.properties["hydrophobic_count"] = int(plif[2])
            rec.add_provenance("plif")
        except DockingError as e:
            logger.debug(f"PLIF failed for {rec.source_id}: {e}")

    return dataset


def compare_plif(plif1: np.ndarray, plif2: np.ndarray) -> float:
    """Compare two PLIFs using Tanimoto similarity.

    Uses the binary portion (last 4 elements) for comparison.

    Args:
        plif1: First PLIF vector.
        plif2: Second PLIF vector.

    Returns:
        Tanimoto similarity (0 to 1).
    """
    # Use binary part for Tanimoto
    b1 = (np.array(plif1) > 0).astype(float)
    b2 = (np.array(plif2) > 0).astype(float)

    intersection = np.sum(b1 * b2)
    union = np.sum(b1) + np.sum(b2) - intersection

    if union == 0:
        return 1.0  # Both empty = identical
    return float(intersection / union)

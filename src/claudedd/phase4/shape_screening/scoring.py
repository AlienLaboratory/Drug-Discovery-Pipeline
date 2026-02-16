"""Shape similarity scoring functions.

Computes shape-based similarity metrics: Shape Tanimoto, protrusion
distance, and combo scores (shape + pharmacophore).
"""

import logging
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdShapeHelpers

from claudedd.core.exceptions import DockingError

logger = logging.getLogger(__name__)


def _ensure_3d(mol: Chem.Mol) -> Chem.Mol:
    """Ensure molecule has 3D coordinates."""
    if mol is None:
        raise DockingError("Cannot score None molecule")
    if mol.GetNumConformers() == 0:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol, params)
    if mol.GetNumConformers() == 0:
        raise DockingError("Cannot generate 3D coords for scoring")
    return mol


def compute_shape_tanimoto(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    conf_id1: int = 0,
    conf_id2: int = 0,
) -> float:
    """Compute Shape Tanimoto similarity.

    Based on volume overlap between two 3D molecules.
    Range: 0 (no overlap) to 1 (identical shape).

    Args:
        mol1: First molecule.
        mol2: Second molecule.
        conf_id1: Conformer ID for mol1.
        conf_id2: Conformer ID for mol2.

    Returns:
        Shape Tanimoto score.
    """
    mol1 = _ensure_3d(mol1)
    mol2 = _ensure_3d(mol2)

    try:
        dist = rdShapeHelpers.ShapeTanimotoDist(mol1, mol2, confId1=conf_id1, confId2=conf_id2)
        return 1.0 - dist  # Convert distance to similarity
    except Exception as e:
        raise DockingError(f"Shape Tanimoto failed: {e}")


def compute_shape_protrusion(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    conf_id1: int = 0,
    conf_id2: int = 0,
) -> float:
    """Compute Shape Protrusion distance.

    Measures asymmetric shape overlap (how much mol1 protrudes beyond mol2).
    Range: 0 (no protrusion) to 1 (complete mismatch).

    Args:
        mol1: First molecule.
        mol2: Second molecule.
        conf_id1: Conformer ID for mol1.
        conf_id2: Conformer ID for mol2.

    Returns:
        Shape protrusion distance.
    """
    mol1 = _ensure_3d(mol1)
    mol2 = _ensure_3d(mol2)

    try:
        return rdShapeHelpers.ShapeProtrudeDist(
            mol1, mol2, confId1=conf_id1, confId2=conf_id2,
        )
    except Exception as e:
        raise DockingError(f"Shape protrusion failed: {e}")


def compute_pharmacophore_score(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    conf_id1: int = 0,
    conf_id2: int = 0,
) -> float:
    """Compute pharmacophore overlap score using Crippen O3A.

    Uses atomic LogP contributions as a proxy for pharmacophore features.

    Args:
        mol1: First molecule (aligned).
        mol2: Second molecule.
        conf_id1: Conformer ID for mol1.
        conf_id2: Conformer ID for mol2.

    Returns:
        Pharmacophore alignment score (higher = better overlap).
    """
    mol1 = _ensure_3d(mol1)
    mol2 = _ensure_3d(mol2)

    try:
        o3a = rdMolAlign.GetCrippenO3A(
            mol1, mol2, prbCid=conf_id1, refCid=conf_id2,
        )
        return o3a.Score()
    except Exception as e:
        logger.debug(f"Pharmacophore score failed: {e}")
        return 0.0


def compute_combo_score(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    conf_id1: int = 0,
    conf_id2: int = 0,
    shape_weight: float = 0.5,
) -> float:
    """Compute combo score: weighted shape + pharmacophore.

    Similar to OpenEye's TanimotoCombo from ROCS.

    Args:
        mol1: First molecule.
        mol2: Second molecule.
        conf_id1: Conformer ID for mol1.
        conf_id2: Conformer ID for mol2.
        shape_weight: Weight for shape component (pharmacophore = 1 - weight).

    Returns:
        Combined score (0 to ~2, higher is better).
    """
    shape = compute_shape_tanimoto(mol1, mol2, conf_id1, conf_id2)
    pharma = compute_pharmacophore_score(mol1, mol2, conf_id1, conf_id2)

    # Normalize pharmacophore score to 0-1 range if possible
    # O3A scores can vary; we'll use them directly for now
    combo = shape_weight * shape + (1.0 - shape_weight) * min(pharma, 1.0)
    return combo

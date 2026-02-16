"""Shape-based molecular alignment using RDKit's Open3DAlign.

Provides O3A and Crippen O3A alignment methods for overlaying
3D conformers and computing alignment scores.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdShapeHelpers

from claudedd.core.exceptions import DockingError

logger = logging.getLogger(__name__)


def _ensure_3d(mol: Chem.Mol, name: str = "molecule") -> Chem.Mol:
    """Ensure molecule has 3D coordinates."""
    if mol is None:
        raise DockingError(f"Cannot align None {name}")
    if mol.GetNumConformers() == 0:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol, params)
        if result == -1:
            raise DockingError(f"Cannot generate 3D coords for {name}")
    return mol


def align_molecules_o3a(
    probe: Chem.Mol,
    reference: Chem.Mol,
    conf_id_probe: int = 0,
    conf_id_ref: int = 0,
) -> Tuple[float, Chem.Mol]:
    """Align probe to reference using Open3DAlign (O3A).

    Args:
        probe: Probe molecule to align.
        reference: Reference molecule (template).
        conf_id_probe: Probe conformer ID.
        conf_id_ref: Reference conformer ID.

    Returns:
        Tuple of (O3A score, aligned probe molecule).
    """
    probe = _ensure_3d(probe, "probe")
    reference = _ensure_3d(reference, "reference")

    try:
        o3a = rdMolAlign.GetO3A(
            probe, reference,
            prbCid=conf_id_probe, refCid=conf_id_ref,
        )
        score = o3a.Align()
        return score, probe
    except Exception as e:
        raise DockingError(f"O3A alignment failed: {e}")


def align_molecules_crippen(
    probe: Chem.Mol,
    reference: Chem.Mol,
    conf_id_probe: int = 0,
    conf_id_ref: int = 0,
) -> Tuple[float, Chem.Mol]:
    """Align using Crippen O3A (based on LogP contributions).

    Uses atomic LogP contributions for pharmacophore-like alignment.

    Args:
        probe: Probe molecule.
        reference: Reference molecule.
        conf_id_probe: Probe conformer ID.
        conf_id_ref: Reference conformer ID.

    Returns:
        Tuple of (alignment score, aligned probe).
    """
    probe = _ensure_3d(probe, "probe")
    reference = _ensure_3d(reference, "reference")

    try:
        o3a = rdMolAlign.GetCrippenO3A(
            probe, reference,
            prbCid=conf_id_probe, refCid=conf_id_ref,
        )
        score = o3a.Align()
        return score, probe
    except Exception as e:
        raise DockingError(f"Crippen O3A alignment failed: {e}")


def compute_rmsd(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    conf_id1: int = 0,
    conf_id2: int = 0,
) -> float:
    """Compute RMSD between two molecules/conformers.

    Args:
        mol1: First molecule.
        mol2: Second molecule.
        conf_id1: Conformer ID for mol1.
        conf_id2: Conformer ID for mol2.

    Returns:
        RMSD in Angstroms.
    """
    mol1 = _ensure_3d(mol1, "mol1")
    mol2 = _ensure_3d(mol2, "mol2")

    try:
        return rdMolAlign.GetBestRMS(mol1, mol2, conf_id1, conf_id2)
    except Exception as e:
        raise DockingError(f"RMSD computation failed: {e}")


def align_to_best_conformer(
    probe: Chem.Mol,
    reference: Chem.Mol,
    conf_id_ref: int = 0,
) -> Tuple[int, float, Chem.Mol]:
    """Align probe's best conformer to reference.

    Tries all probe conformers and returns the one with best O3A score.

    Args:
        probe: Probe molecule with multiple conformers.
        reference: Reference molecule.
        conf_id_ref: Reference conformer ID.

    Returns:
        Tuple of (best_conf_id, best_score, aligned_probe).
    """
    probe = _ensure_3d(probe, "probe")
    reference = _ensure_3d(reference, "reference")

    if probe.GetNumConformers() == 0:
        raise DockingError("Probe has no conformers")

    best_score = -1.0
    best_conf_id = -1

    for conf in probe.GetConformers():
        cid = conf.GetId()
        try:
            o3a = rdMolAlign.GetO3A(
                probe, reference,
                prbCid=cid, refCid=conf_id_ref,
            )
            score = o3a.Score()
            if score > best_score:
                best_score = score
                best_conf_id = cid
        except Exception:
            continue

    if best_conf_id == -1:
        raise DockingError("No conformers could be aligned")

    # Do the final alignment with the best conformer
    o3a = rdMolAlign.GetO3A(
        probe, reference,
        prbCid=best_conf_id, refCid=conf_id_ref,
    )
    o3a.Align()

    return best_conf_id, best_score, probe

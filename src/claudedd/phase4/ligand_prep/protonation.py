"""Tautomer enumeration and protonation state handling.

Uses RDKit's MolStandardize for tautomer enumeration and
canonical tautomer selection.
"""

import logging
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

from claudedd.core.exceptions import DockingError

logger = logging.getLogger(__name__)


def enumerate_tautomers(
    mol: Chem.Mol,
    max_tautomers: int = 10,
) -> List[Chem.Mol]:
    """Enumerate tautomers of a molecule.

    Args:
        mol: Input molecule.
        max_tautomers: Maximum number of tautomers to generate.

    Returns:
        List of tautomer molecules (includes the input).
    """
    if mol is None:
        raise DockingError("Cannot enumerate tautomers for None molecule")

    try:
        enumerator = rdMolStandardize.TautomerEnumerator()
        tautomers = list(enumerator.Enumerate(mol))
        return tautomers[:max_tautomers]
    except Exception as e:
        logger.warning(f"Tautomer enumeration failed: {e}")
        return [mol]


def get_dominant_tautomer(mol: Chem.Mol) -> Chem.Mol:
    """Get the canonical (dominant) tautomer.

    Args:
        mol: Input molecule.

    Returns:
        Canonical tautomer molecule.
    """
    if mol is None:
        raise DockingError("Cannot get tautomer for None molecule")

    try:
        enumerator = rdMolStandardize.TautomerEnumerator()
        return enumerator.Canonicalize(mol)
    except Exception as e:
        logger.warning(f"Canonical tautomer failed: {e}")
        return mol


def add_hydrogens_3d(mol: Chem.Mol) -> Chem.Mol:
    """Add explicit hydrogens with 3D coordinates.

    If the molecule already has 3D coordinates, hydrogens are placed
    using the existing geometry. Otherwise, generates 3D coordinates first.

    Args:
        mol: Input molecule.

    Returns:
        Molecule with explicit hydrogens and 3D coordinates.
    """
    if mol is None:
        raise DockingError("Cannot add hydrogens to None molecule")

    mol = Chem.AddHs(mol)

    if mol.GetNumConformers() == 0:
        # Generate 3D coordinates
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol, params)
        if result == -1:
            params.useRandomCoords = True
            AllChem.EmbedMolecule(mol, params)

    return mol

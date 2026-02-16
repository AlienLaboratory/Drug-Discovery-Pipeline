"""3D molecular geometry measurements.

Distance, angle, dihedral, volume, and bounding box computations
for molecules with 3D coordinates.
"""

import logging
from typing import Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms, rdShapeHelpers

from drugflow.core.exceptions import DockingError

logger = logging.getLogger(__name__)


def _get_position(mol: Chem.Mol, atom_idx: int, conf_id: int = 0) -> np.ndarray:
    """Get atom position as numpy array."""
    conf = mol.GetConformer(conf_id)
    pos = conf.GetAtomPosition(atom_idx)
    return np.array([pos.x, pos.y, pos.z])


def measure_distance(
    mol: Chem.Mol,
    atom_i: int,
    atom_j: int,
    conf_id: int = 0,
) -> float:
    """Measure distance between two atoms.

    Args:
        mol: Molecule with 3D conformer.
        atom_i: First atom index.
        atom_j: Second atom index.
        conf_id: Conformer ID.

    Returns:
        Distance in Angstroms.
    """
    if mol is None or mol.GetNumConformers() == 0:
        raise DockingError("Molecule has no 3D conformers")
    return rdMolTransforms.GetBondLength(mol.GetConformer(conf_id), atom_i, atom_j)


def measure_angle(
    mol: Chem.Mol,
    i: int,
    j: int,
    k: int,
    conf_id: int = 0,
) -> float:
    """Measure angle between three atoms (i-j-k).

    Args:
        mol: Molecule with 3D conformer.
        i, j, k: Atom indices (j is the vertex).
        conf_id: Conformer ID.

    Returns:
        Angle in degrees.
    """
    if mol is None or mol.GetNumConformers() == 0:
        raise DockingError("Molecule has no 3D conformers")
    return rdMolTransforms.GetAngleDeg(mol.GetConformer(conf_id), i, j, k)


def measure_dihedral(
    mol: Chem.Mol,
    i: int,
    j: int,
    k: int,
    l: int,
    conf_id: int = 0,
) -> float:
    """Measure dihedral angle between four atoms.

    Args:
        mol: Molecule with 3D conformer.
        i, j, k, l: Atom indices.
        conf_id: Conformer ID.

    Returns:
        Dihedral angle in degrees (-180 to 180).
    """
    if mol is None or mol.GetNumConformers() == 0:
        raise DockingError("Molecule has no 3D conformers")
    return rdMolTransforms.GetDihedralDeg(mol.GetConformer(conf_id), i, j, k, l)


def compute_molecular_volume(
    mol: Chem.Mol,
    conf_id: int = 0,
) -> float:
    """Compute approximate molecular volume.

    Uses RDKit's grid-based volume estimation.

    Args:
        mol: Molecule with 3D conformer.
        conf_id: Conformer ID.

    Returns:
        Volume in cubic Angstroms.
    """
    if mol is None or mol.GetNumConformers() == 0:
        raise DockingError("Molecule has no 3D conformers")

    try:
        # Use self-overlap as volume approximation
        vol = AllChem.ComputeMolVolume(mol, confId=conf_id)
        return vol
    except Exception as e:
        raise DockingError(f"Volume computation failed: {e}")


def compute_bounding_box(
    mol: Chem.Mol,
    conf_id: int = 0,
    padding: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the bounding box of a molecule.

    Args:
        mol: Molecule with 3D conformer.
        conf_id: Conformer ID.
        padding: Extra padding around the box (Angstroms).

    Returns:
        Tuple of (min_corner, max_corner) as numpy arrays.
    """
    if mol is None or mol.GetNumConformers() == 0:
        raise DockingError("Molecule has no 3D conformers")

    conf = mol.GetConformer(conf_id)
    positions = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        positions.append([pos.x, pos.y, pos.z])

    positions = np.array(positions)
    min_corner = positions.min(axis=0) - padding
    max_corner = positions.max(axis=0) + padding

    return min_corner, max_corner

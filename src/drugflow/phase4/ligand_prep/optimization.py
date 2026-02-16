"""Force field energy minimization for conformers.

Supports MMFF94 and UFF force fields for geometry optimization,
strain energy computation, and 3D structure validation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from drugflow.core.constants import OPTIMIZATION_MAX_ITERS
from drugflow.core.exceptions import DockingError

logger = logging.getLogger(__name__)


def optimize_conformer(
    mol: Chem.Mol,
    conf_id: int = -1,
    force_field: str = "MMFF",
    max_iters: int = OPTIMIZATION_MAX_ITERS,
) -> Tuple[Chem.Mol, float]:
    """Optimize a single conformer using a force field.

    Args:
        mol: Molecule with 3D conformer(s).
        conf_id: Conformer ID to optimize (-1 for first).
        force_field: "MMFF" or "UFF".
        max_iters: Maximum optimization iterations.

    Returns:
        Tuple of (optimized molecule, final energy).
    """
    if mol is None or mol.GetNumConformers() == 0:
        raise DockingError("Molecule has no 3D conformers to optimize")

    if conf_id == -1:
        conf_id = mol.GetConformers()[0].GetId()

    try:
        if force_field == "MMFF":
            result = AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=max_iters)
        else:
            result = AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=max_iters)

        # Get final energy
        if force_field == "MMFF":
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)

        energy = ff.CalcEnergy() if ff is not None else float("inf")
        return mol, energy
    except Exception as e:
        raise DockingError(f"Optimization failed: {e}")


def optimize_all_conformers(
    mol: Chem.Mol,
    force_field: str = "MMFF",
    max_iters: int = OPTIMIZATION_MAX_ITERS,
) -> List[Tuple[int, float]]:
    """Optimize all conformers and return energies.

    Args:
        mol: Molecule with multiple conformers.
        force_field: Force field type.
        max_iters: Max iterations per conformer.

    Returns:
        List of (conformer_id, energy) tuples sorted by energy.
    """
    if mol is None or mol.GetNumConformers() == 0:
        raise DockingError("Molecule has no conformers")

    results = []
    if force_field == "MMFF":
        opt_results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=max_iters)
        for i, (converged, energy) in enumerate(opt_results):
            conf_id = mol.GetConformers()[i].GetId()
            results.append((conf_id, energy))
    else:
        opt_results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=max_iters)
        for i, (converged, energy) in enumerate(opt_results):
            conf_id = mol.GetConformers()[i].GetId()
            results.append((conf_id, energy))

    results.sort(key=lambda x: x[1])
    return results


def compute_strain_energy(
    mol: Chem.Mol,
    conf_id: int = -1,
    force_field: str = "MMFF",
) -> float:
    """Compute strain energy relative to lowest-energy conformer.

    Args:
        mol: Molecule with conformers.
        conf_id: Conformer to compute strain for.
        force_field: Force field type.

    Returns:
        Strain energy (kcal/mol) = E(conf) - E(min).
    """
    from drugflow.phase4.ligand_prep.conformers import get_conformer_energies

    energies = get_conformer_energies(mol, force_field)
    if not energies:
        raise DockingError("No energies computed")

    min_energy = min(energies)

    if conf_id == -1:
        conf_id = mol.GetConformers()[0].GetId()

    conf_ids = [c.GetId() for c in mol.GetConformers()]
    if conf_id not in conf_ids:
        raise DockingError(f"Conformer {conf_id} not found")

    idx = conf_ids.index(conf_id)
    return energies[idx] - min_energy


def validate_3d_geometry(
    mol: Chem.Mol,
    conf_id: int = -1,
) -> Dict[str, Any]:
    """Validate 3D geometry of a conformer.

    Checks for reasonable bond lengths, detects clashes,
    and verifies the molecule has 3D coordinates.

    Args:
        mol: Molecule with conformer.
        conf_id: Conformer ID.

    Returns:
        Dict with validation results.
    """
    if mol is None or mol.GetNumConformers() == 0:
        return {"valid": False, "reason": "no conformers"}

    if conf_id == -1:
        conf_id = mol.GetConformers()[0].GetId()

    conf = mol.GetConformer(conf_id)
    n_atoms = mol.GetNumAtoms()

    # Check 3D coordinates are present
    positions = []
    for i in range(n_atoms):
        pos = conf.GetAtomPosition(i)
        positions.append(np.array([pos.x, pos.y, pos.z]))

    positions = np.array(positions)

    # Check all coords are not zero (2D artifact)
    all_zero_z = np.all(np.abs(positions[:, 2]) < 0.01)

    # Check for atomic clashes (atoms < 0.5 Å apart)
    n_clashes = 0
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 0.5:
                n_clashes += 1

    # Check bond lengths
    bond_issues = 0
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        dist = np.linalg.norm(positions[i] - positions[j])
        if dist < 0.5 or dist > 5.0:
            bond_issues += 1

    valid = not all_zero_z and n_clashes == 0 and bond_issues == 0

    return {
        "valid": valid,
        "is_3d": not all_zero_z,
        "n_atoms": n_atoms,
        "n_clashes": n_clashes,
        "bond_issues": bond_issues,
        "conf_id": conf_id,
    }

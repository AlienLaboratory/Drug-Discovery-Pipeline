"""Multi-conformer generation using RDKit's ETKDGv3.

Generates 3D conformers for small molecules, computes energies,
and prunes redundant conformers by RMSD.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from drugflow.core.constants import (
    CONFORMER_MAX_ATTEMPTS,
    CONFORMER_NUM_CONFS,
    CONFORMER_PRUNE_RMSD,
)
from drugflow.core.exceptions import DockingError
from drugflow.core.models import MoleculeDataset, MoleculeRecord

logger = logging.getLogger(__name__)


def generate_conformers(
    mol: Chem.Mol,
    n_confs: int = CONFORMER_NUM_CONFS,
    prune_rmsd: float = CONFORMER_PRUNE_RMSD,
    max_attempts: int = CONFORMER_MAX_ATTEMPTS,
    seed: int = 42,
) -> Chem.Mol:
    """Generate 3D conformers using ETKDGv3.

    Args:
        mol: Input 2D or 3D molecule.
        n_confs: Number of conformers to generate.
        prune_rmsd: RMSD threshold for pruning (Angstroms).
        max_attempts: Maximum embedding attempts.
        seed: Random seed.

    Returns:
        Molecule with embedded conformers.

    Raises:
        DockingError: If molecule is None or embedding fails.
    """
    if mol is None:
        raise DockingError("Cannot generate conformers for None molecule")

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.maxIterations = max_attempts
    params.pruneRmsThresh = prune_rmsd
    params.numThreads = 0  # Use all available threads

    n_embedded = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)

    if n_embedded == 0:
        # Fallback: try with less strict parameters
        params.useRandomCoords = True
        n_embedded = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)

    if n_embedded == 0:
        raise DockingError("Failed to generate any conformers")

    logger.debug(f"Generated {n_embedded} conformers")
    return mol


def get_conformer_energies(
    mol: Chem.Mol,
    force_field: str = "MMFF",
) -> List[float]:
    """Get energy of each conformer.

    Args:
        mol: Molecule with conformers.
        force_field: "MMFF" or "UFF".

    Returns:
        List of energies (kcal/mol), one per conformer.
    """
    if mol is None or mol.GetNumConformers() == 0:
        raise DockingError("Molecule has no conformers")

    energies = []
    for conf in mol.GetConformers():
        conf_id = conf.GetId()
        try:
            if force_field == "MMFF":
                props = AllChem.MMFFGetMoleculeProperties(mol)
                if props is None:
                    ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                else:
                    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)

            if ff is not None:
                energies.append(ff.CalcEnergy())
            else:
                energies.append(float("inf"))
        except Exception:
            energies.append(float("inf"))

    return energies


def get_lowest_energy_conformer(
    mol: Chem.Mol,
    force_field: str = "MMFF",
) -> Tuple[int, float]:
    """Find the lowest-energy conformer.

    Args:
        mol: Molecule with conformers.
        force_field: Force field for energy calculation.

    Returns:
        Tuple of (conformer_id, energy).
    """
    energies = get_conformer_energies(mol, force_field)
    if not energies:
        raise DockingError("No conformer energies computed")

    conf_ids = [conf.GetId() for conf in mol.GetConformers()]
    min_idx = int(np.argmin(energies))
    return conf_ids[min_idx], energies[min_idx]


def prune_conformers(
    mol: Chem.Mol,
    rmsd_threshold: float = CONFORMER_PRUNE_RMSD,
) -> Chem.Mol:
    """Remove redundant conformers by RMSD clustering.

    Args:
        mol: Molecule with conformers.
        rmsd_threshold: Minimum RMSD between kept conformers.

    Returns:
        Molecule with pruned conformer set.
    """
    if mol is None or mol.GetNumConformers() <= 1:
        return mol

    conf_ids = [conf.GetId() for conf in mol.GetConformers()]
    keep = [conf_ids[0]]

    for i in range(1, len(conf_ids)):
        is_unique = True
        for kept_id in keep:
            try:
                rmsd = rdMolAlign.GetBestRMS(mol, mol, kept_id, conf_ids[i])
                if rmsd < rmsd_threshold:
                    is_unique = False
                    break
            except Exception:
                continue
        if is_unique:
            keep.append(conf_ids[i])

    # Remove conformers not in keep list
    to_remove = [cid for cid in conf_ids if cid not in keep]
    for cid in sorted(to_remove, reverse=True):
        mol.RemoveConformer(cid)

    logger.debug(f"Pruned to {mol.GetNumConformers()} conformers (from {len(conf_ids)})")
    return mol


def generate_conformers_dataset(
    dataset: MoleculeDataset,
    n_confs: int = CONFORMER_NUM_CONFS,
    prune_rmsd: float = CONFORMER_PRUNE_RMSD,
    seed: int = 42,
) -> MoleculeDataset:
    """Generate conformers for all valid molecules in a dataset.

    Args:
        dataset: Input dataset.
        n_confs: Number of conformers per molecule.
        prune_rmsd: RMSD pruning threshold.
        seed: Random seed.

    Returns:
        Dataset with conformers embedded in mol objects.
    """
    for rec in dataset.valid_records:
        if rec.mol is None:
            continue
        try:
            rec.mol = generate_conformers(
                rec.mol, n_confs=n_confs, prune_rmsd=prune_rmsd, seed=seed,
            )
            n_conf = rec.mol.GetNumConformers()
            rec.metadata["n_conformers"] = n_conf
            rec.add_provenance(f"conformers:{n_conf}")
        except DockingError as e:
            logger.warning(f"Conformer generation failed for {rec.source_id}: {e}")
            rec.metadata["n_conformers"] = 0

    return dataset

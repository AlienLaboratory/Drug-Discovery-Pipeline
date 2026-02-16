"""End-to-end ligand preparation pipeline.

Combines conformer generation, optimization, protonation, and export
into a single preparation workflow.
"""

import logging
import os
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem

from drugflow.core.constants import CONFORMER_NUM_CONFS, CONFORMER_PRUNE_RMSD
from drugflow.core.exceptions import DockingError
from drugflow.core.models import MoleculeDataset, MoleculeRecord
from drugflow.phase4.ligand_prep.conformers import (
    generate_conformers,
    get_lowest_energy_conformer,
    prune_conformers,
)
from drugflow.phase4.ligand_prep.optimization import optimize_all_conformers
from drugflow.phase4.ligand_prep.protonation import add_hydrogens_3d, get_dominant_tautomer

logger = logging.getLogger(__name__)


def prepare_ligand(
    mol: Chem.Mol,
    n_confs: int = CONFORMER_NUM_CONFS,
    optimize: bool = True,
    force_field: str = "MMFF",
    prune_rmsd: float = CONFORMER_PRUNE_RMSD,
    seed: int = 42,
) -> Chem.Mol:
    """Full ligand preparation pipeline.

    Steps: canonical tautomer → add Hs → generate conformers → optimize → prune.

    Args:
        mol: Input molecule (2D or 3D).
        n_confs: Number of conformers to generate.
        optimize: Whether to energy-minimize conformers.
        force_field: "MMFF" or "UFF".
        prune_rmsd: RMSD threshold for pruning.
        seed: Random seed.

    Returns:
        Prepared molecule with optimized 3D conformers.
    """
    if mol is None:
        raise DockingError("Cannot prepare None molecule")

    # Step 1: Canonical tautomer
    mol = get_dominant_tautomer(mol)

    # Step 2: Generate conformers (adds Hs internally)
    mol = generate_conformers(mol, n_confs=n_confs, prune_rmsd=prune_rmsd, seed=seed)

    # Step 3: Optimize
    if optimize and mol.GetNumConformers() > 0:
        try:
            optimize_all_conformers(mol, force_field=force_field)
        except DockingError:
            logger.warning("Optimization failed, using unoptimized conformers")

    # Step 4: Prune again after optimization
    mol = prune_conformers(mol, rmsd_threshold=prune_rmsd)

    return mol


def prepare_ligand_dataset(
    dataset: MoleculeDataset,
    n_confs: int = 20,
    optimize: bool = True,
    force_field: str = "MMFF",
    seed: int = 42,
) -> MoleculeDataset:
    """Prepare all ligands in a dataset.

    Args:
        dataset: Input dataset.
        n_confs: Conformers per molecule.
        optimize: Whether to optimize.
        force_field: Force field.
        seed: Random seed.

    Returns:
        Dataset with prepared 3D molecules.
    """
    for rec in dataset.valid_records:
        if rec.mol is None:
            continue
        try:
            rec.mol = prepare_ligand(
                rec.mol, n_confs=n_confs, optimize=optimize,
                force_field=force_field, seed=seed,
            )
            n_conf = rec.mol.GetNumConformers()
            rec.metadata["n_conformers"] = n_conf
            rec.metadata["prepared"] = True

            # Store lowest energy
            try:
                best_id, best_energy = get_lowest_energy_conformer(rec.mol, force_field)
                rec.properties["lowest_energy"] = best_energy
                rec.properties["best_conformer_id"] = best_id
            except Exception:
                pass

            rec.add_provenance("ligand_prep")
        except DockingError as e:
            logger.warning(f"Ligand prep failed for {rec.source_id}: {e}")
            rec.metadata["prepared"] = False

    return dataset


def export_ligand_pdb(
    mol: Chem.Mol,
    output_path: str,
    conf_id: int = 0,
) -> str:
    """Export a ligand conformer to PDB format.

    Args:
        mol: Prepared molecule with 3D coordinates.
        output_path: Output file path.
        conf_id: Conformer ID to export.

    Returns:
        Path to written file.
    """
    if mol is None or mol.GetNumConformers() == 0:
        raise DockingError("Molecule has no 3D conformers to export")

    pdb_block = Chem.MolToPDBBlock(mol, confId=conf_id)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(pdb_block)
    return output_path


def export_ligand_sdf(
    mol: Chem.Mol,
    output_path: str,
) -> str:
    """Export all conformers to SDF format.

    Args:
        mol: Prepared molecule with conformers.
        output_path: Output file path.

    Returns:
        Path to written file.
    """
    if mol is None or mol.GetNumConformers() == 0:
        raise DockingError("Molecule has no conformers to export")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = Chem.SDWriter(output_path)
    for conf in mol.GetConformers():
        writer.write(mol, confId=conf.GetId())
    writer.close()
    return output_path

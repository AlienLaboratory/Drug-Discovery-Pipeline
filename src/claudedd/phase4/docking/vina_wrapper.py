"""AutoDock Vina wrapper (optional).

Provides molecular docking via AutoDock Vina Python bindings.
Requires: pip install vina meeko
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from rdkit import Chem

from claudedd.core.constants import VINA_ENERGY_RANGE, VINA_EXHAUSTIVENESS, VINA_NUM_MODES
from claudedd.core.exceptions import DockingError
from claudedd.core.models import MoleculeDataset, MoleculeRecord
from claudedd.phase4.docking.grid import DockingBox

logger = logging.getLogger(__name__)


def _check_vina_available() -> bool:
    """Check if AutoDock Vina Python bindings are installed."""
    try:
        from vina import Vina
        return True
    except ImportError:
        return False


def _check_meeko_available() -> bool:
    """Check if Meeko (Vina prep tool) is installed."""
    try:
        import meeko
        return True
    except ImportError:
        return False


def dock_vina(
    ligand_mol: Chem.Mol,
    protein_path: str,
    box: DockingBox,
    exhaustiveness: int = VINA_EXHAUSTIVENESS,
    n_modes: int = VINA_NUM_MODES,
    energy_range: float = VINA_ENERGY_RANGE,
) -> List[Dict]:
    """Dock a ligand against a protein using AutoDock Vina.

    Requires: pip install vina meeko

    Args:
        ligand_mol: Prepared ligand with 3D conformer.
        protein_path: Path to prepared protein PDBQT file.
        box: Docking search box.
        exhaustiveness: Vina exhaustiveness parameter.
        n_modes: Number of binding modes to return.
        energy_range: Energy range for modes (kcal/mol).

    Returns:
        List of docking result dicts with score, coords, etc.

    Raises:
        DockingError: If Vina is not installed or docking fails.
    """
    if not _check_vina_available():
        raise DockingError(
            "AutoDock Vina is not installed. "
            "Install with: pip install vina meeko"
        )

    if ligand_mol is None or ligand_mol.GetNumConformers() == 0:
        raise DockingError("Ligand has no 3D conformers for docking")

    if not os.path.exists(protein_path):
        raise DockingError(f"Protein file not found: {protein_path}")

    try:
        from vina import Vina

        v = Vina(sf_name="vina")
        v.set_receptor(protein_path)

        # Write ligand to temp PDBQT
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
            pdb_block = Chem.MolToPDBBlock(ligand_mol)
            f.write(pdb_block)
            lig_pdb_path = f.name

        # Set up docking
        v.set_ligand_from_file(lig_pdb_path)
        v.compute_vina_maps(
            center=[box.center_x, box.center_y, box.center_z],
            box_size=[box.size_x, box.size_y, box.size_z],
        )

        v.dock(
            exhaustiveness=exhaustiveness,
            n_poses=n_modes,
        )

        # Parse results
        energies = v.energies()
        results = []
        for i, energy_row in enumerate(energies):
            results.append({
                "pose_id": i,
                "vina_score": float(energy_row[0]),
                "inter_energy": float(energy_row[1]) if len(energy_row) > 1 else None,
                "intra_energy": float(energy_row[2]) if len(energy_row) > 2 else None,
            })

        # Clean up
        os.unlink(lig_pdb_path)

        return results
    except ImportError:
        raise DockingError("Vina not available. Install with: pip install vina")
    except Exception as e:
        raise DockingError(f"Docking failed: {e}")


def dock_dataset_vina(
    dataset: MoleculeDataset,
    protein_path: str,
    box: DockingBox,
    exhaustiveness: int = VINA_EXHAUSTIVENESS,
    n_modes: int = VINA_NUM_MODES,
) -> MoleculeDataset:
    """Dock all molecules in a dataset using Vina.

    Args:
        dataset: Dataset of prepared ligands.
        protein_path: Path to protein PDBQT.
        box: Docking search box.
        exhaustiveness: Vina exhaustiveness.
        n_modes: Number of binding modes.

    Returns:
        Dataset with docking scores in properties.
    """
    if not _check_vina_available():
        raise DockingError(
            "AutoDock Vina is not installed. "
            "Install with: pip install vina meeko"
        )

    for rec in dataset.valid_records:
        if rec.mol is None or rec.mol.GetNumConformers() == 0:
            continue
        try:
            results = dock_vina(
                rec.mol, protein_path, box,
                exhaustiveness=exhaustiveness, n_modes=n_modes,
            )
            if results:
                rec.properties["vina_score"] = results[0]["vina_score"]
                rec.properties["vina_poses"] = len(results)
                rec.metadata["docking_results"] = results
            rec.add_provenance("docking:vina")
        except DockingError as e:
            logger.warning(f"Docking failed for {rec.source_id}: {e}")

    return dataset

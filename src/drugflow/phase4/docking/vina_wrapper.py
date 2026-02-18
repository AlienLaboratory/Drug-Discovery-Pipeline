"""AutoDock Vina wrapper — Python bindings OR CLI binary.

Supports two modes:
1. Python bindings: ``pip install vina meeko`` (requires Boost on Windows)
2. CLI binary: vina executable found on PATH or in project ``tools/`` dir

The wrapper auto-detects which mode is available and prefers the Python
bindings when both exist.  Ligand PDBQT conversion uses Meeko; protein
PDBQT conversion is handled by a lightweight built-in converter.
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

from drugflow.core.constants import (
    VINA_DEFAULT_SCORING,
    VINA_ENERGY_RANGE,
    VINA_EXECUTABLE_NAMES,
    VINA_EXHAUSTIVENESS,
    VINA_NUM_MODES,
)
from drugflow.core.exceptions import DockingError
from drugflow.core.models import MoleculeDataset, MoleculeRecord
from drugflow.phase4.docking.grid import DockingBox

logger = logging.getLogger(__name__)

# ── Availability checks ─────────────────────────────────────────────

def _check_vina_python() -> bool:
    """Check if AutoDock Vina **Python bindings** are installed."""
    try:
        from vina import Vina  # noqa: F401
        return True
    except ImportError:
        return False


def _check_vina_binary() -> Optional[str]:
    """Find the Vina CLI executable.

    Search order:
    1. ``VINA_EXECUTABLE`` environment variable
    2. System PATH
    3. ``<project_root>/tools/`` directory

    Returns:
        Absolute path to the Vina binary, or None.
    """
    # 1. Environment variable
    env_path = os.environ.get("VINA_EXECUTABLE")
    if env_path and os.path.isfile(env_path):
        return os.path.abspath(env_path)

    # 2. System PATH
    for name in VINA_EXECUTABLE_NAMES:
        found = shutil.which(name)
        if found:
            return os.path.abspath(found)

    # 3. Project tools/ directory (walk up from this file)
    module_dir = Path(__file__).resolve().parent
    for ancestor in [module_dir] + list(module_dir.parents):
        tools_dir = ancestor / "tools"
        if tools_dir.is_dir():
            for name in VINA_EXECUTABLE_NAMES:
                candidate = tools_dir / name
                if candidate.is_file():
                    return str(candidate)

    return None


def _check_vina_available() -> bool:
    """Check if *any* usable Vina back-end is available."""
    return _check_vina_python() or (_check_vina_binary() is not None)


def _check_meeko_available() -> bool:
    """Check if Meeko (Vina molecule-preparation tool) is installed."""
    try:
        from meeko import MoleculePreparation  # noqa: F401
        return True
    except ImportError:
        return False


def get_vina_backend() -> str:
    """Return the best available Vina back-end.

    Returns:
        ``"python"`` | ``"binary"`` | ``"none"``
    """
    if _check_vina_python():
        return "python"
    if _check_vina_binary() is not None:
        return "binary"
    return "none"

# ── PDBQT conversion utilities ──────────────────────────────────────

# AutoDock atom-type mapping (element → AD4 type)
_AD4_TYPES = {
    "C": "C", "N": "N", "O": "OA", "S": "SA", "H": "HD",
    "F": "F", "Cl": "Cl", "Br": "Br", "I": "I",
    "P": "P", "Fe": "Fe", "Zn": "Zn", "Mg": "Mg",
    "Mn": "Mn", "Ca": "Ca", "Na": "NA",
}


def mol_to_pdbqt(mol: Chem.Mol) -> str:
    """Convert an RDKit Mol (with 3-D coords) to PDBQT string via Meeko.

    Meeko handles torsion-tree construction, Gasteiger charges, and
    AutoDock atom types properly.

    Args:
        mol: RDKit molecule **with Hs and at least one conformer**.

    Returns:
        PDBQT string.

    Raises:
        DockingError: On conversion failure or missing Meeko.
    """
    if not _check_meeko_available():
        raise DockingError(
            "Meeko is required for ligand PDBQT conversion. "
            "Install with: pip install meeko"
        )
    if mol is None or mol.GetNumConformers() == 0:
        raise DockingError("Molecule has no 3D conformers for PDBQT conversion")

    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy

        prep = MoleculePreparation()
        mol_setups = prep.prepare(mol)
        pdbqt_str, is_ok, err = PDBQTWriterLegacy.write_string(mol_setups[0])
        if not is_ok:
            raise DockingError(f"Meeko PDBQT write failed: {err}")
        return pdbqt_str
    except DockingError:
        raise
    except Exception as e:
        raise DockingError(f"Ligand PDBQT conversion failed: {e}")


def protein_pdb_to_pdbqt(pdb_path: str, output_path: Optional[str] = None) -> str:
    """Convert a protein PDB file to PDBQT format.

    Uses a lightweight conversion: copies ATOM/HETATM lines and appends
    AutoDock atom types + zero partial charges.  This is equivalent to
    ``prepare_receptor4.py -A hydrogens`` for rigid docking.

    Args:
        pdb_path: Path to cleaned protein PDB file.
        output_path: Where to write the PDBQT. If None, writes next to
            the PDB with ``.pdbqt`` extension.

    Returns:
        Path to the written PDBQT file.
    """
    if not os.path.isfile(pdb_path):
        raise DockingError(f"PDB file not found: {pdb_path}")

    if output_path is None:
        output_path = os.path.splitext(pdb_path)[0] + ".pdbqt"

    pdbqt_lines = []
    with open(pdb_path, "r") as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                # PDB format: columns 1-66  (+  columns 67-80 for PDBQT)
                # We need: original 66 chars + partial_charge (8.3f) + AD_type (2s)
                raw = line.rstrip("\n")

                # Element symbol (columns 77-78 in standard PDB)
                element = raw[76:78].strip() if len(raw) >= 78 else ""
                if not element:
                    # Fallback: derive from atom name (cols 13-16)
                    atom_name = raw[12:16].strip()
                    element = re.sub(r"\d", "", atom_name)[:2].strip()
                    if len(element) == 2:
                        element = element[0].upper() + element[1].lower()
                    else:
                        element = element.upper()

                ad_type = _AD4_TYPES.get(element, "C")

                # Build PDBQT line: 66 chars + charge(8) + type(2) + newline
                padded = raw[:54].ljust(54)
                # Occupancy + B-factor columns (55-66) then charge + type
                occ_bfac = raw[54:66] if len(raw) >= 66 else "  1.00  0.00"
                occ_bfac = occ_bfac.ljust(12)
                pdbqt_line = f"{padded}{occ_bfac}    {0.000:+7.3f} {ad_type:<2s}"
                pdbqt_lines.append(pdbqt_line)
            elif line.startswith(("TER", "END")):
                pdbqt_lines.append(line.rstrip("\n"))

    with open(output_path, "w") as fh:
        fh.write("\n".join(pdbqt_lines) + "\n")

    logger.info("Protein PDBQT written: %s (%d lines)", output_path, len(pdbqt_lines))
    return output_path

# ── Docking via Python bindings ─────────────────────────────────────

def _dock_vina_python(
    ligand_mol: Chem.Mol,
    protein_pdbqt: str,
    box: DockingBox,
    exhaustiveness: int,
    n_modes: int,
    energy_range: float,
) -> List[Dict]:
    """Dock one ligand using Vina *Python* bindings."""
    from vina import Vina

    v = Vina(sf_name="vina")
    v.set_receptor(protein_pdbqt)

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
        pdb_block = Chem.MolToPDBBlock(ligand_mol)
        f.write(pdb_block)
        lig_pdb = f.name

    try:
        v.set_ligand_from_file(lig_pdb)
        v.compute_vina_maps(
            center=[box.center_x, box.center_y, box.center_z],
            box_size=[box.size_x, box.size_y, box.size_z],
        )
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_modes)

        energies = v.energies()
        results = []
        for i, row in enumerate(energies):
            results.append({
                "pose_id": i,
                "vina_score": float(row[0]),
                "inter_energy": float(row[1]) if len(row) > 1 else None,
                "intra_energy": float(row[2]) if len(row) > 2 else None,
            })
        return results
    finally:
        os.unlink(lig_pdb)

# ── Docking via CLI binary ──────────────────────────────────────────

def _parse_vina_output(stdout: str) -> List[Dict]:
    """Parse Vina CLI stdout for docking scores.

    Vina prints a table like::

        mode |   affinity   | dist from best mode
             | (kcal/mol)   | rmsd l.b.| rmsd u.b.
        -----+--------------+----------+----------
           1       -8.3          0.000      0.000
           2       -7.9          1.234      2.345
    """
    results = []
    in_table = False
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("-----+"):
            in_table = True
            continue
        if in_table and stripped and stripped[0].isdigit():
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    pose_id = int(parts[0]) - 1  # 0-indexed
                    score = float(parts[1])
                    rmsd_lb = float(parts[2]) if len(parts) > 2 else None
                    rmsd_ub = float(parts[3]) if len(parts) > 3 else None
                    results.append({
                        "pose_id": pose_id,
                        "vina_score": score,
                        "rmsd_lb": rmsd_lb,
                        "rmsd_ub": rmsd_ub,
                    })
                except ValueError:
                    continue
        elif in_table and not stripped:
            break  # end of table

    return results


def _dock_vina_cli(
    ligand_pdbqt: str,
    protein_pdbqt: str,
    box: DockingBox,
    exhaustiveness: int,
    n_modes: int,
    energy_range: float,
    vina_path: str,
    output_path: Optional[str] = None,
    scoring: str = VINA_DEFAULT_SCORING,
) -> List[Dict]:
    """Dock one ligand using Vina CLI binary.

    Args:
        ligand_pdbqt: Path to ligand PDBQT file.
        protein_pdbqt: Path to protein PDBQT file.
        box: Docking search box.
        exhaustiveness: Search exhaustiveness.
        n_modes: Max binding modes.
        energy_range: Energy range (kcal/mol).
        vina_path: Path to Vina executable.
        output_path: Optional output PDBQT for docked poses.
        scoring: Scoring function name.

    Returns:
        List of result dicts with vina_score, pose_id, etc.
    """
    cmd = [
        vina_path,
        "--receptor", protein_pdbqt,
        "--ligand", ligand_pdbqt,
        "--center_x", str(box.center_x),
        "--center_y", str(box.center_y),
        "--center_z", str(box.center_z),
        "--size_x", str(box.size_x),
        "--size_y", str(box.size_y),
        "--size_z", str(box.size_z),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(n_modes),
        "--energy_range", str(energy_range),
        "--scoring", scoring,
        "--verbosity", "1",
    ]

    if output_path:
        cmd.extend(["--out", output_path])

    logger.debug("Vina CLI: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min per ligand
        )
    except FileNotFoundError:
        raise DockingError(f"Vina binary not found at: {vina_path}")
    except subprocess.TimeoutExpired:
        raise DockingError("Vina docking timed out (>10 min)")

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise DockingError(f"Vina exited with code {proc.returncode}: {stderr}")

    results = _parse_vina_output(proc.stdout)
    if not results:
        logger.warning("No docking poses found in Vina output")

    return results

# ── Public high-level API ───────────────────────────────────────────

def dock_vina(
    ligand_mol: Chem.Mol,
    protein_path: str,
    box: DockingBox,
    exhaustiveness: int = VINA_EXHAUSTIVENESS,
    n_modes: int = VINA_NUM_MODES,
    energy_range: float = VINA_ENERGY_RANGE,
    output_dir: Optional[str] = None,
) -> List[Dict]:
    """Dock a ligand against a protein using AutoDock Vina.

    Automatically selects the best available backend:
    - Python bindings (``from vina import Vina``) if installed
    - CLI binary (``vina.exe`` on PATH or in ``tools/``) otherwise

    Args:
        ligand_mol: RDKit Mol with at least one 3-D conformer.
        protein_path: Path to protein PDB **or** PDBQT file.
        box: Docking search box.
        exhaustiveness: Search thoroughness (default 8).
        n_modes: Number of binding modes (default 9).
        energy_range: Max energy difference (kcal/mol, default 3.0).
        output_dir: Optional directory for intermediate files.

    Returns:
        List of docking result dicts (pose_id, vina_score, …).

    Raises:
        DockingError: If no Vina back-end is available or docking fails.
    """
    backend = get_vina_backend()

    if backend == "none":
        raise DockingError(
            "AutoDock Vina is not available. Either:\n"
            "  1. Install Python bindings: pip install vina meeko\n"
            "  2. Place vina executable in tools/ or on PATH\n"
            "  3. Set VINA_EXECUTABLE env var"
        )

    if ligand_mol is None or ligand_mol.GetNumConformers() == 0:
        raise DockingError("Ligand has no 3D conformers for docking")

    if not os.path.exists(protein_path):
        raise DockingError(f"Protein file not found: {protein_path}")

    # Ensure protein is PDBQT
    protein_pdbqt = protein_path
    if protein_path.lower().endswith(".pdb"):
        if output_dir:
            protein_pdbqt = os.path.join(
                output_dir, os.path.basename(protein_path).replace(".pdb", ".pdbqt")
            )
        else:
            protein_pdbqt = protein_path.replace(".pdb", ".pdbqt")

        if not os.path.exists(protein_pdbqt):
            protein_pdbqt = protein_pdb_to_pdbqt(protein_path, protein_pdbqt)
            logger.info("Converted protein to PDBQT: %s", protein_pdbqt)

    try:
        if backend == "python":
            return _dock_vina_python(
                ligand_mol, protein_pdbqt, box,
                exhaustiveness, n_modes, energy_range,
            )
        else:
            # CLI binary path
            vina_bin = _check_vina_binary()

            # Write ligand PDBQT via Meeko
            has_hs = any(a.GetAtomicNum() == 1 for a in ligand_mol.GetAtoms())
            pdbqt_str = mol_to_pdbqt(ligand_mol if has_hs else Chem.AddHs(ligand_mol))

            tmpdir = output_dir or tempfile.gettempdir()
            lig_pdbqt = os.path.join(tmpdir, "_lig_temp.pdbqt")
            with open(lig_pdbqt, "w") as f:
                f.write(pdbqt_str)

            try:
                results = _dock_vina_cli(
                    lig_pdbqt, protein_pdbqt, box,
                    exhaustiveness, n_modes, energy_range,
                    vina_bin,
                )
                return results
            finally:
                if os.path.exists(lig_pdbqt) and not output_dir:
                    os.unlink(lig_pdbqt)

    except DockingError:
        raise
    except Exception as e:
        raise DockingError(f"Docking failed: {e}")


def dock_dataset_vina(
    dataset: MoleculeDataset,
    protein_path: str,
    box: DockingBox,
    exhaustiveness: int = VINA_EXHAUSTIVENESS,
    n_modes: int = VINA_NUM_MODES,
    output_dir: Optional[str] = None,
) -> MoleculeDataset:
    """Dock all molecules in a dataset using Vina.

    Converts protein to PDBQT once, then iterates over ligands.

    Args:
        dataset: Dataset of prepared ligands with 3-D conformers.
        protein_path: Path to protein PDB or PDBQT.
        box: Docking search box.
        exhaustiveness: Vina exhaustiveness.
        n_modes: Number of binding modes.
        output_dir: Directory for PDBQT intermediates and poses.

    Returns:
        Same dataset with ``vina_score`` and ``vina_poses`` in properties.
    """
    backend = get_vina_backend()
    if backend == "none":
        raise DockingError(
            "AutoDock Vina is not available. Either:\n"
            "  1. Install Python bindings: pip install vina meeko\n"
            "  2. Place vina executable in tools/ or on PATH\n"
            "  3. Set VINA_EXECUTABLE env var"
        )

    logger.info("Docking %d molecules via Vina (%s)", len(dataset.valid_records), backend)

    # Pre-convert protein to PDBQT if needed
    protein_pdbqt = protein_path
    if protein_path.lower().endswith(".pdb"):
        if output_dir:
            protein_pdbqt = os.path.join(
                output_dir, os.path.basename(protein_path).replace(".pdb", ".pdbqt")
            )
        else:
            protein_pdbqt = protein_path.replace(".pdb", ".pdbqt")
        if not os.path.exists(protein_pdbqt):
            protein_pdbqt = protein_pdb_to_pdbqt(protein_path, protein_pdbqt)

    vina_bin = _check_vina_binary() if backend == "binary" else None
    docked = 0
    failed = 0

    for i, rec in enumerate(dataset.valid_records):
        if rec.mol is None or rec.mol.GetNumConformers() == 0:
            continue

        try:
            if backend == "python":
                results = _dock_vina_python(
                    rec.mol, protein_pdbqt, box,
                    exhaustiveness, n_modes, VINA_ENERGY_RANGE,
                )
            else:
                has_hs = any(a.GetAtomicNum() == 1 for a in rec.mol.GetAtoms())
                mol_h = rec.mol if has_hs else Chem.AddHs(rec.mol)
                # Re-embed if Hs were added but no conf
                if mol_h.GetNumConformers() == 0:
                    AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())

                pdbqt_str = mol_to_pdbqt(mol_h)
                tmpdir = output_dir or tempfile.gettempdir()
                lig_file = os.path.join(tmpdir, f"_lig_{rec.record_id or i}.pdbqt")
                out_file = os.path.join(tmpdir, f"_out_{rec.record_id or i}.pdbqt") if output_dir else None

                with open(lig_file, "w") as f:
                    f.write(pdbqt_str)

                results = _dock_vina_cli(
                    lig_file, protein_pdbqt, box,
                    exhaustiveness, n_modes, VINA_ENERGY_RANGE,
                    vina_bin, output_path=out_file,
                )

                # Clean up temp ligand if not saving
                if not output_dir and os.path.exists(lig_file):
                    os.unlink(lig_file)

            if results:
                rec.properties["vina_score"] = results[0]["vina_score"]
                rec.properties["vina_poses"] = len(results)
                rec.metadata["docking_results"] = results
                docked += 1
            rec.add_provenance("docking:vina")

        except DockingError as e:
            logger.warning("Docking failed for %s: %s", rec.record_id or f"mol_{i}", e)
            failed += 1
        except Exception as e:
            logger.warning("Unexpected error docking %s: %s", rec.record_id or f"mol_{i}", e)
            failed += 1

    logger.info("Docking complete: %d docked, %d failed", docked, failed)
    return dataset

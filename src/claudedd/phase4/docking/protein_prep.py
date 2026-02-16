"""Protein preparation for docking (optional BioPython).

Loads, cleans, and prepares protein structures from PDB files.
Falls back to RDKit-only methods when BioPython is not available.
"""

import logging
from typing import List, Optional

from rdkit import Chem

from claudedd.core.exceptions import DockingError

logger = logging.getLogger(__name__)


def _check_biopython_available() -> bool:
    """Check if BioPython is installed."""
    try:
        from Bio.PDB import PDBParser
        return True
    except ImportError:
        return False


def load_protein_pdb(path: str) -> Chem.Mol:
    """Load protein from PDB file using RDKit.

    Args:
        path: Path to PDB file.

    Returns:
        RDKit molecule object.

    Raises:
        DockingError: If loading fails.
    """
    try:
        mol = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False)
        if mol is None:
            raise DockingError(f"Failed to parse PDB file: {path}")
        return mol
    except Exception as e:
        raise DockingError(f"Failed to load protein PDB: {e}")


def remove_water(protein_mol: Chem.Mol) -> Chem.Mol:
    """Remove water molecules from protein.

    Args:
        protein_mol: Protein molecule (may contain water HOH/WAT).

    Returns:
        Protein without water molecules.
    """
    if protein_mol is None:
        raise DockingError("Cannot process None protein")

    try:
        # Remove water by filtering out oxygen-only residues
        rw_mol = Chem.RWMol(protein_mol)
        atoms_to_remove = []

        for atom in rw_mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            if info is not None:
                resname = info.GetResidueName().strip()
                if resname in ("HOH", "WAT", "H2O", "DOD"):
                    atoms_to_remove.append(atom.GetIdx())

        # Remove in reverse order to preserve indices
        for idx in sorted(atoms_to_remove, reverse=True):
            rw_mol.RemoveAtom(idx)

        return rw_mol.GetMol()
    except Exception as e:
        logger.warning(f"Water removal failed: {e}")
        return protein_mol


def remove_heteroatoms(
    protein_mol: Chem.Mol,
    keep_list: Optional[List[str]] = None,
) -> Chem.Mol:
    """Remove heteroatom residues (ligands, ions, etc.) from protein.

    Args:
        protein_mol: Protein molecule.
        keep_list: List of residue names to keep (e.g., ["ZN", "MG"]).

    Returns:
        Protein without heteroatom residues.
    """
    if protein_mol is None:
        raise DockingError("Cannot process None protein")

    if keep_list is None:
        keep_list = []

    # Standard amino acid residue names
    standard_residues = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
        "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
        "THR", "TRP", "TYR", "VAL",
    }

    try:
        rw_mol = Chem.RWMol(protein_mol)
        atoms_to_remove = []

        for atom in rw_mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            if info is not None:
                resname = info.GetResidueName().strip()
                if resname not in standard_residues and resname not in keep_list:
                    atoms_to_remove.append(atom.GetIdx())

        for idx in sorted(atoms_to_remove, reverse=True):
            rw_mol.RemoveAtom(idx)

        return rw_mol.GetMol()
    except Exception as e:
        logger.warning(f"Heteroatom removal failed: {e}")
        return protein_mol


def prepare_protein(
    pdb_path: str,
    remove_waters: bool = True,
    remove_hets: bool = True,
    keep_list: Optional[List[str]] = None,
) -> Chem.Mol:
    """End-to-end protein preparation.

    Args:
        pdb_path: Path to PDB file.
        remove_waters: Whether to remove water molecules.
        remove_hets: Whether to remove heteroatom residues.
        keep_list: Residue names to keep if removing heteroatoms.

    Returns:
        Cleaned protein molecule.
    """
    protein = load_protein_pdb(pdb_path)

    if remove_waters:
        protein = remove_water(protein)

    if remove_hets:
        protein = remove_heteroatoms(protein, keep_list=keep_list)

    logger.info(f"Protein prepared: {protein.GetNumAtoms()} atoms")
    return protein

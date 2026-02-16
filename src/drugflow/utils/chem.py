"""Shared RDKit utility functions."""

from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import inchi as rdInchi
from rdkit.Chem.Scaffolds import MurckoScaffold


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def mol_to_smiles(mol: Chem.Mol, canonical: bool = True) -> str:
    return Chem.MolToSmiles(mol, canonical=canonical)


def mol_to_inchi(mol: Chem.Mol) -> str:
    return rdInchi.MolToInchi(mol) or ""


def mol_to_inchikey(mol: Chem.Mol) -> str:
    inchi_str = rdInchi.MolToInchi(mol)
    if inchi_str:
        return rdInchi.InchiToInchiKey(inchi_str) or ""
    return ""


def add_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    return Chem.AddHs(mol)


def remove_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    return Chem.RemoveHs(mol)


def compute_2d_coords(mol: Chem.Mol) -> Chem.Mol:
    AllChem.Compute2DCoords(mol)
    return mol


def compute_3d_coords(
    mol: Chem.Mol,
    optimize: bool = True,
    random_seed: int = 42,
) -> Chem.Mol:
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    AllChem.EmbedMolecule(mol, params)
    if optimize:
        AllChem.MMFFOptimizeMolecule(mol)
    return mol


def get_scaffold(mol: Chem.Mol) -> Chem.Mol:
    return MurckoScaffold.GetScaffoldForMol(mol)

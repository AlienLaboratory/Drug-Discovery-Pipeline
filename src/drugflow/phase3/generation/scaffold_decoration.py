"""Scaffold decoration for de novo molecular generation.

Extracts Murcko scaffolds from seed molecules and decorates attachment
points with common medicinal chemistry R-groups.
"""

import logging
import random
from itertools import product as itertools_product
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, RWMol
from rdkit.Chem.Scaffolds import MurckoScaffold

from drugflow.core.constants import R_GROUP_LIBRARY
from drugflow.core.exceptions import GenerationError
from drugflow.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus
from drugflow.utils.chem import get_scaffold, mol_to_smiles, smiles_to_mol

logger = logging.getLogger(__name__)


def extract_core_scaffold(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Extract the Murcko scaffold from a molecule.

    Args:
        mol: Input RDKit molecule.

    Returns:
        Scaffold molecule, or None if extraction fails.
    """
    if mol is None:
        return None
    try:
        return get_scaffold(mol)
    except Exception:
        return None


def find_decoration_sites(mol: Chem.Mol) -> List[int]:
    """Find atoms suitable for R-group decoration.

    Identifies non-ring atoms with available valence (hydrogen-bearing atoms
    on the periphery). These are attachment points for R-group addition.

    Args:
        mol: Input molecule (typically a scaffold).

    Returns:
        List of atom indices that can accept decorations.
    """
    if mol is None:
        return []

    sites = []
    mol_h = Chem.AddHs(mol)

    for atom in mol_h.GetAtoms():
        # Skip hydrogens
        if atom.GetAtomicNum() == 1:
            continue
        # Check if atom has attached hydrogens
        n_h = sum(
            1 for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() == 1
        )
        if n_h > 0:
            sites.append(atom.GetIdx())

    return sites


def get_r_group_library() -> List[Chem.Mol]:
    """Get library of common medicinal chemistry R-groups as mol objects.

    Returns:
        List of RDKit molecule objects for R-groups.
    """
    r_groups = []
    for smi in R_GROUP_LIBRARY:
        mol = smiles_to_mol(smi)
        if mol is not None:
            r_groups.append(mol)
    return r_groups


def _attach_r_group(
    mol: Chem.Mol,
    atom_idx: int,
    r_group_smi: str,
) -> Optional[Chem.Mol]:
    """Attach an R-group at a specific atom position.

    Replaces one hydrogen at the target atom with the R-group.

    Args:
        mol: Parent molecule.
        atom_idx: Atom index to attach R-group to.
        r_group_smi: SMILES of the R-group to attach.

    Returns:
        New molecule with R-group attached, or None if attachment fails.
    """
    try:
        # Use reaction SMARTS for generic attachment
        # [*:1][H] >> [*:1][R]
        # Simpler approach: combine and bond
        r_mol = Chem.MolFromSmiles(r_group_smi)
        if r_mol is None:
            return None

        combo = Chem.RWMol(Chem.CombineMols(mol, r_mol))
        r_group_start_idx = mol.GetNumAtoms()

        # Find a hydrogen on the target atom to replace
        mol_h = Chem.AddHs(mol)
        target_atom = mol_h.GetAtomWithIdx(atom_idx)
        has_h = any(
            n.GetAtomicNum() == 1
            for n in target_atom.GetNeighbors()
        )
        if not has_h:
            return None

        # Add bond between target atom and first atom of R-group
        combo.AddBond(atom_idx, r_group_start_idx, Chem.BondType.SINGLE)

        try:
            Chem.SanitizeMol(combo)
            return combo.GetMol()
        except Exception:
            return None
    except Exception:
        return None


def decorate_scaffold(
    scaffold: Chem.Mol,
    r_groups: Optional[List[str]] = None,
    max_molecules: int = 500,
    max_sites: int = 3,
    seed: int = 42,
) -> List[Chem.Mol]:
    """Decorate a scaffold with R-groups at available positions.

    For each decoration site, tries attaching each R-group and collects
    valid products.

    Args:
        scaffold: Scaffold molecule to decorate.
        r_groups: List of R-group SMILES. Uses default library if None.
        max_molecules: Maximum number of decorated molecules.
        max_sites: Maximum number of sites to decorate simultaneously.
        seed: Random seed.

    Returns:
        List of valid decorated molecules.
    """
    if scaffold is None:
        raise GenerationError("Cannot decorate None scaffold")

    rng = random.Random(seed)

    if r_groups is None:
        r_groups = R_GROUP_LIBRARY

    sites = find_decoration_sites(scaffold)
    if not sites:
        logger.warning("No decoration sites found on scaffold")
        return []

    # Limit sites to max_sites
    if len(sites) > max_sites:
        sites = rng.sample(sites, max_sites)

    products: List[Chem.Mol] = []
    seen: Set[str] = set()
    scaffold_smi = Chem.MolToSmiles(scaffold, canonical=True)
    seen.add(scaffold_smi)

    # Single-site decoration
    for site in sites:
        for r_smi in r_groups:
            if len(products) >= max_molecules:
                break
            new_mol = _attach_r_group(scaffold, site, r_smi)
            if new_mol is not None:
                try:
                    smi = Chem.MolToSmiles(new_mol, canonical=True)
                    if smi not in seen:
                        seen.add(smi)
                        products.append(new_mol)
                except Exception:
                    continue

    # Multi-site decoration (pairs of sites)
    if len(sites) >= 2 and len(products) < max_molecules:
        for i, site1 in enumerate(sites):
            for site2 in sites[i + 1:]:
                rg_pairs = [
                    (r1, r2)
                    for r1 in rng.sample(r_groups, min(5, len(r_groups)))
                    for r2 in rng.sample(r_groups, min(5, len(r_groups)))
                ]
                for r1_smi, r2_smi in rg_pairs:
                    if len(products) >= max_molecules:
                        break
                    mol1 = _attach_r_group(scaffold, site1, r1_smi)
                    if mol1 is not None:
                        mol2 = _attach_r_group(mol1, site2, r2_smi)
                        if mol2 is not None:
                            try:
                                smi = Chem.MolToSmiles(mol2, canonical=True)
                                if smi not in seen:
                                    seen.add(smi)
                                    products.append(mol2)
                            except Exception:
                                continue

    return products


def generate_from_scaffold(
    seed_mol: Chem.Mol,
    n_molecules: int = 100,
    r_groups: Optional[List[str]] = None,
    seed: int = 42,
) -> MoleculeDataset:
    """End-to-end scaffold-based generation from a seed molecule.

    Extracts the scaffold, finds decoration sites, and generates variants
    by attaching R-groups.

    Args:
        seed_mol: Seed molecule to derive scaffold from.
        n_molecules: Target number of molecules to generate.
        r_groups: Optional custom R-group library.
        seed: Random seed.

    Returns:
        MoleculeDataset of generated molecules.
    """
    if seed_mol is None:
        raise GenerationError("Seed molecule cannot be None")

    scaffold = extract_core_scaffold(seed_mol)
    if scaffold is None:
        raise GenerationError("Could not extract scaffold from seed molecule")

    scaffold_smi = Chem.MolToSmiles(scaffold, canonical=True)
    logger.info(f"Scaffold: {scaffold_smi}")

    products = decorate_scaffold(
        scaffold,
        r_groups=r_groups,
        max_molecules=n_molecules,
        seed=seed,
    )

    seed_smi = Chem.MolToSmiles(seed_mol, canonical=True)
    records = []
    for i, mol in enumerate(products):
        smi = Chem.MolToSmiles(mol, canonical=True)
        rec = MoleculeRecord(
            mol=mol,
            source_id=f"scaffold_{i}",
            smiles=smi,
            status=MoleculeStatus.RAW,
        )
        rec.add_provenance("generated:scaffold_decoration")
        rec.metadata["generation_method"] = "scaffold_decoration"
        rec.metadata["parent_scaffold"] = scaffold_smi
        rec.metadata["is_novel"] = smi != seed_smi
        records.append(rec)

    result = MoleculeDataset(
        records=records,
        name=f"scaffold_generated_{len(records)}",
    )
    logger.info(f"Scaffold decoration complete: {len(records)} molecules")
    return result

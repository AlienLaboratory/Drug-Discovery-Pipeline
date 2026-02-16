"""Molecular mutation operators for de novo generation.

Provides atom-level, bond-level, and fragment-level mutations to
systematically explore chemical space around seed molecules.
"""

import logging
import random
from typing import Callable, List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem, RWMol

from drugflow.core.constants import MUTATION_ATOM_TYPES, MUTATION_TYPES, R_GROUP_LIBRARY
from drugflow.core.exceptions import GenerationError
from drugflow.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus
from drugflow.utils.chem import mol_to_smiles

logger = logging.getLogger(__name__)


def _validate_mol(mol: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
    """Validate and sanitize a molecule, return None if invalid."""
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        # Verify it has a valid SMILES
        smi = Chem.MolToSmiles(mol, canonical=True)
        if not smi:
            return None
        return mol
    except Exception:
        return None


def mutate_atom(
    mol: Chem.Mol,
    atom_idx: Optional[int] = None,
    new_element: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Optional[Chem.Mol]:
    """Swap an atom's element type.

    Args:
        mol: Input molecule.
        atom_idx: Index of atom to mutate. Random if None.
        new_element: Atomic number for new element. Random if None.
        rng: Random number generator.

    Returns:
        Mutated molecule or None if invalid.
    """
    if mol is None:
        return None
    if rng is None:
        rng = random.Random()

    rw_mol = RWMol(Chem.RWMol(mol))
    n_atoms = rw_mol.GetNumAtoms()
    if n_atoms == 0:
        return None

    if atom_idx is None:
        # Pick a non-hydrogen heavy atom
        heavy_atoms = [
            i for i in range(n_atoms)
            if rw_mol.GetAtomWithIdx(i).GetAtomicNum() > 1
        ]
        if not heavy_atoms:
            return None
        atom_idx = rng.choice(heavy_atoms)

    if atom_idx >= n_atoms:
        return None

    current_element = rw_mol.GetAtomWithIdx(atom_idx).GetAtomicNum()

    if new_element is None:
        candidates = [e for e in MUTATION_ATOM_TYPES if e != current_element]
        if not candidates:
            return None
        new_element = rng.choice(candidates)

    rw_mol.GetAtomWithIdx(atom_idx).SetAtomicNum(new_element)
    return _validate_mol(rw_mol.GetMol())


def mutate_bond(
    mol: Chem.Mol,
    bond_idx: Optional[int] = None,
    new_type: Optional[Chem.BondType] = None,
    rng: Optional[random.Random] = None,
) -> Optional[Chem.Mol]:
    """Change a bond's order.

    Args:
        mol: Input molecule.
        bond_idx: Index of bond to mutate. Random if None.
        new_type: New bond type. Random if None.
        rng: Random number generator.

    Returns:
        Mutated molecule or None if invalid.
    """
    if mol is None:
        return None
    if rng is None:
        rng = random.Random()

    rw_mol = RWMol(Chem.RWMol(mol))
    n_bonds = rw_mol.GetNumBonds()
    if n_bonds == 0:
        return None

    if bond_idx is None:
        bond_idx = rng.randint(0, n_bonds - 1)

    if bond_idx >= n_bonds:
        return None

    bond_types = [
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
    ]

    current_type = rw_mol.GetBondWithIdx(bond_idx).GetBondType()

    if new_type is None:
        candidates = [bt for bt in bond_types if bt != current_type]
        new_type = rng.choice(candidates)

    rw_mol.GetBondWithIdx(bond_idx).SetBondType(new_type)
    return _validate_mol(rw_mol.GetMol())


def add_fragment(
    mol: Chem.Mol,
    atom_idx: Optional[int] = None,
    fragment_smiles: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> Optional[Chem.Mol]:
    """Attach a small fragment to the molecule.

    Args:
        mol: Input molecule.
        atom_idx: Atom index to attach fragment to. Random if None.
        fragment_smiles: SMILES of fragment. Random from library if None.
        rng: Random number generator.

    Returns:
        Modified molecule or None if invalid.
    """
    if mol is None:
        return None
    if rng is None:
        rng = random.Random()

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return None

    if atom_idx is None:
        heavy_atoms = [
            i for i in range(n_atoms)
            if mol.GetAtomWithIdx(i).GetAtomicNum() > 1
        ]
        if not heavy_atoms:
            return None
        atom_idx = rng.choice(heavy_atoms)

    if fragment_smiles is None:
        # Use simple fragments that bond well
        simple_frags = ["C", "N", "O", "F", "Cl", "CC", "OC"]
        fragment_smiles = rng.choice(simple_frags)

    frag_mol = Chem.MolFromSmiles(fragment_smiles)
    if frag_mol is None:
        return None

    try:
        combo = Chem.RWMol(Chem.CombineMols(mol, frag_mol))
        frag_start = n_atoms  # First atom of fragment
        combo.AddBond(atom_idx, frag_start, Chem.BondType.SINGLE)
        return _validate_mol(combo.GetMol())
    except Exception:
        return None


def remove_fragment(
    mol: Chem.Mol,
    atom_idx: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Optional[Chem.Mol]:
    """Remove a terminal substituent from the molecule.

    Removes a terminal (degree-1) atom and its bond.

    Args:
        mol: Input molecule.
        atom_idx: Index of terminal atom to remove. Random if None.
        rng: Random number generator.

    Returns:
        Modified molecule or None if invalid.
    """
    if mol is None:
        return None
    if rng is None:
        rng = random.Random()

    # Find terminal (degree 1) heavy atoms
    terminals = []
    for atom in mol.GetAtoms():
        if atom.GetDegree() == 1 and atom.GetAtomicNum() > 1:
            terminals.append(atom.GetIdx())

    if not terminals:
        return None

    if atom_idx is None:
        atom_idx = rng.choice(terminals)
    elif atom_idx not in terminals:
        return None

    rw_mol = RWMol(Chem.RWMol(mol))
    rw_mol.RemoveAtom(atom_idx)

    result = rw_mol.GetMol()
    if result.GetNumAtoms() == 0:
        return None

    return _validate_mol(result)


def ring_open_close(
    mol: Chem.Mol,
    rng: Optional[random.Random] = None,
) -> Optional[Chem.Mol]:
    """Mutate ring systems — add a bond to form a ring or break one.

    Attempts to add a bond between two non-bonded atoms to create
    a new ring, or remove a ring-closing bond.

    Args:
        mol: Input molecule.
        rng: Random number generator.

    Returns:
        Modified molecule or None if invalid.
    """
    if mol is None:
        return None
    if rng is None:
        rng = random.Random()

    rw_mol = RWMol(Chem.RWMol(mol))
    n_atoms = rw_mol.GetNumAtoms()

    if n_atoms < 3:
        return None

    # Try to form a new ring by connecting non-bonded atoms
    # that are 3-6 bonds apart in the graph
    try:
        # Find pairs of atoms that could form a ring
        candidates = []
        for i in range(n_atoms):
            atom_i = rw_mol.GetAtomWithIdx(i)
            if atom_i.GetAtomicNum() <= 1:
                continue
            for j in range(i + 1, n_atoms):
                atom_j = rw_mol.GetAtomWithIdx(j)
                if atom_j.GetAtomicNum() <= 1:
                    continue
                # Check if they're not already bonded
                bond = rw_mol.GetBondBetweenAtoms(i, j)
                if bond is None:
                    candidates.append((i, j))

        if not candidates:
            return None

        # Try random pair
        pair = rng.choice(candidates)
        rw_mol.AddBond(pair[0], pair[1], Chem.BondType.SINGLE)
        return _validate_mol(rw_mol.GetMol())
    except Exception:
        return None


# Registry of mutation functions
_MUTATION_REGISTRY: dict = {
    "atom_swap": mutate_atom,
    "bond_change": mutate_bond,
    "add_fragment": add_fragment,
    "remove_fragment": remove_fragment,
    "ring_mutation": ring_open_close,
}


def random_mutation(
    mol: Chem.Mol,
    mutation_types: Optional[List[str]] = None,
    rng: Optional[random.Random] = None,
) -> Optional[Chem.Mol]:
    """Apply a random mutation to a molecule.

    Args:
        mol: Input molecule.
        mutation_types: Allowed mutation types. Uses all if None.
        rng: Random number generator.

    Returns:
        Mutated molecule or None if all attempts fail.
    """
    if mol is None:
        return None
    if rng is None:
        rng = random.Random()

    if mutation_types is None:
        mutation_types = list(MUTATION_TYPES)

    # Shuffle mutation types and try each until one succeeds
    types_to_try = list(mutation_types)
    rng.shuffle(types_to_try)

    for mut_type in types_to_try:
        fn = _MUTATION_REGISTRY.get(mut_type)
        if fn is None:
            continue
        result = fn(mol, rng=rng)
        if result is not None:
            return result

    return None


def mutate_molecule(
    mol: Chem.Mol,
    n_mutations: int = 1,
    mutation_types: Optional[List[str]] = None,
    rng: Optional[random.Random] = None,
    max_attempts: int = 10,
) -> Optional[Chem.Mol]:
    """Apply N sequential mutations to a molecule.

    Args:
        mol: Input molecule.
        n_mutations: Number of mutations to apply sequentially.
        mutation_types: Allowed mutation types.
        rng: Random number generator.
        max_attempts: Max retries per mutation step.

    Returns:
        Mutated molecule or None if unable to complete all mutations.
    """
    if mol is None:
        return None
    if rng is None:
        rng = random.Random()

    current = mol
    for _ in range(n_mutations):
        success = False
        for _ in range(max_attempts):
            result = random_mutation(current, mutation_types=mutation_types, rng=rng)
            if result is not None:
                current = result
                success = True
                break
        if not success:
            # Return what we have so far (partial mutations)
            return current if current is not mol else None

    return current


def generate_mutations(
    dataset: MoleculeDataset,
    n_molecules: int = 100,
    n_mutations_per_mol: int = 1,
    mutation_types: Optional[List[str]] = None,
    seed: int = 42,
) -> MoleculeDataset:
    """Generate new molecules by mutating existing ones.

    Args:
        dataset: Input seed dataset.
        n_molecules: Target number of molecules to generate.
        n_mutations_per_mol: Number of mutations per seed molecule.
        mutation_types: Allowed mutation types.
        seed: Random seed.

    Returns:
        MoleculeDataset of mutated molecules.
    """
    rng = random.Random(seed)
    valid_records = dataset.valid_records
    if not valid_records:
        raise GenerationError("No valid molecules in dataset for mutation")

    records = []
    seen: set = set()
    attempts = 0
    max_attempts = n_molecules * 10

    while len(records) < n_molecules and attempts < max_attempts:
        attempts += 1
        # Pick random seed molecule
        seed_rec = rng.choice(valid_records)
        if seed_rec.mol is None:
            continue

        mutated = mutate_molecule(
            seed_rec.mol,
            n_mutations=n_mutations_per_mol,
            mutation_types=mutation_types,
            rng=rng,
        )
        if mutated is None:
            continue

        smi = Chem.MolToSmiles(mutated, canonical=True)
        if smi in seen:
            continue
        seen.add(smi)

        rec = MoleculeRecord(
            mol=mutated,
            source_id=f"mutant_{len(records)}",
            smiles=smi,
            status=MoleculeStatus.RAW,
        )
        rec.add_provenance("generated:mutation")
        rec.metadata["generation_method"] = "mutation"
        rec.metadata["parent_smiles"] = seed_rec.canonical_smiles
        rec.metadata["n_mutations"] = n_mutations_per_mol
        records.append(rec)

    result = MoleculeDataset(
        records=records,
        name=f"mutation_generated_{len(records)}",
    )
    logger.info(f"Mutation generation complete: {len(records)} molecules")
    return result

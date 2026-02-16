"""BRICS fragment decomposition and recombination for de novo generation.

Uses RDKit's BRICS (Breaking of Retrosynthetically Interesting Chemical
Substructures) to decompose molecules into fragments and recombine them
to create novel molecules.
"""

import logging
import random
from collections import Counter
from typing import Dict, List, Optional, Set

from rdkit import Chem
from rdkit.Chem import BRICS

from claudedd.core.constants import (
    BRICS_DEFAULT_MAX_DEPTH,
    BRICS_MAX_BUILD_MOLECULES,
    BRICS_MAX_FRAGMENTS,
    BRICS_MIN_FRAGMENT_FREQUENCY,
)
from claudedd.core.exceptions import GenerationError
from claudedd.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus
from claudedd.utils.chem import mol_to_smiles, smiles_to_mol

logger = logging.getLogger(__name__)


def decompose_molecule(mol: Chem.Mol) -> List[str]:
    """Decompose a molecule into BRICS fragments.

    Args:
        mol: RDKit molecule object.

    Returns:
        List of fragment SMILES strings.

    Raises:
        GenerationError: If molecule is None or decomposition fails.
    """
    if mol is None:
        raise GenerationError("Cannot decompose None molecule")
    try:
        fragments = list(BRICS.BRICSDecompose(mol))
        return fragments
    except Exception as e:
        raise GenerationError(f"BRICS decomposition failed: {e}")


def decompose_dataset(
    dataset: MoleculeDataset,
) -> Dict[str, List[str]]:
    """Decompose all valid molecules in a dataset into BRICS fragments.

    Args:
        dataset: Input dataset of molecules.

    Returns:
        Dict mapping source SMILES to list of fragment SMILES.
    """
    result: Dict[str, List[str]] = {}
    for rec in dataset.valid_records:
        if rec.mol is None:
            continue
        try:
            frags = decompose_molecule(rec.mol)
            smi = rec.canonical_smiles or mol_to_smiles(rec.mol)
            result[smi] = frags
        except GenerationError:
            logger.debug(f"Skipping molecule {rec.source_id}: decomposition failed")
    return result


def build_fragment_library(
    dataset: MoleculeDataset,
    min_frequency: int = BRICS_MIN_FRAGMENT_FREQUENCY,
    max_fragments: int = BRICS_MAX_FRAGMENTS,
) -> List[str]:
    """Build a curated fragment library from a dataset.

    Fragments that appear in at least `min_frequency` molecules are kept.

    Args:
        dataset: Input dataset.
        min_frequency: Minimum number of molecules a fragment must appear in.
        max_fragments: Maximum fragments to return.

    Returns:
        Sorted list of fragment SMILES.
    """
    fragment_counts: Counter = Counter()
    decompositions = decompose_dataset(dataset)

    for frags in decompositions.values():
        # Count unique fragments per molecule
        fragment_counts.update(set(frags))

    # Filter by frequency and limit
    library = [
        frag for frag, count in fragment_counts.most_common()
        if count >= min_frequency
    ]
    return library[:max_fragments]


def enumerate_from_fragments(
    fragments: List[str],
    max_molecules: int = BRICS_MAX_BUILD_MOLECULES,
    max_depth: int = BRICS_DEFAULT_MAX_DEPTH,
    seed: int = 42,
) -> List[Chem.Mol]:
    """Enumerate molecules by recombining BRICS fragments.

    Args:
        fragments: List of fragment SMILES (from BRICS decomposition).
        max_molecules: Maximum number of molecules to generate.
        max_depth: Maximum BRICS build depth (higher = larger molecules).
        seed: Random seed for reproducibility.

    Returns:
        List of valid RDKit molecule objects.
    """
    if not fragments:
        raise GenerationError("No fragments provided for enumeration")

    # Convert fragment SMILES to mols
    frag_mols = []
    for smi in fragments:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            frag_mols.append(mol)

    if not frag_mols:
        raise GenerationError("No valid fragments could be parsed")

    try:
        # BRICSBuild returns a generator; collect up to max_molecules
        builder = BRICS.BRICSBuild(frag_mols)
        products = []
        seen: Set[str] = set()

        for mol in builder:
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
                smi = Chem.MolToSmiles(mol, canonical=True)
                if smi not in seen:
                    seen.add(smi)
                    products.append(mol)
                    if len(products) >= max_molecules:
                        break
            except Exception:
                continue

        return products
    except Exception as e:
        raise GenerationError(f"BRICS enumeration failed: {e}")


def recombine_two_molecules(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    max_products: int = 10,
) -> List[Chem.Mol]:
    """Recombine fragments from two molecules to create hybrids.

    Args:
        mol1: First parent molecule.
        mol2: Second parent molecule.
        max_products: Maximum number of hybrid products.

    Returns:
        List of valid hybrid molecule objects.
    """
    if mol1 is None or mol2 is None:
        raise GenerationError("Both parent molecules must be provided")

    try:
        frags1 = list(BRICS.BRICSDecompose(mol1))
        frags2 = list(BRICS.BRICSDecompose(mol2))
    except Exception as e:
        raise GenerationError(f"Failed to decompose parents: {e}")

    combined_frags = list(set(frags1 + frags2))
    if not combined_frags:
        return []

    # Convert to mol objects for BRICSBuild
    frag_mols = []
    for smi in combined_frags:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            frag_mols.append(mol)

    if not frag_mols:
        return []

    try:
        builder = BRICS.BRICSBuild(frag_mols)
        products = []
        seen: Set[str] = set()
        # Exclude parent SMILES
        parent1_smi = Chem.MolToSmiles(mol1, canonical=True)
        parent2_smi = Chem.MolToSmiles(mol2, canonical=True)
        seen.add(parent1_smi)
        seen.add(parent2_smi)

        for mol in builder:
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
                smi = Chem.MolToSmiles(mol, canonical=True)
                if smi not in seen:
                    seen.add(smi)
                    products.append(mol)
                    if len(products) >= max_products:
                        break
            except Exception:
                continue

        return products
    except Exception as e:
        logger.debug(f"BRICS recombination failed: {e}")
        return []


def generate_brics(
    dataset: MoleculeDataset,
    n_molecules: int = 100,
    min_fragment_frequency: int = BRICS_MIN_FRAGMENT_FREQUENCY,
    max_depth: int = BRICS_DEFAULT_MAX_DEPTH,
    seed: int = 42,
) -> MoleculeDataset:
    """End-to-end BRICS-based molecular generation.

    Decomposes the input dataset into fragments, builds a fragment library,
    and enumerates new molecules by recombination.

    Args:
        dataset: Input seed dataset.
        n_molecules: Target number of molecules to generate.
        min_fragment_frequency: Minimum fragment frequency for library.
        max_depth: Maximum BRICS build depth.
        seed: Random seed.

    Returns:
        MoleculeDataset of generated molecules.
    """
    logger.info(f"BRICS generation: building fragment library from {len(dataset)} molecules")

    # Build fragment library
    library = build_fragment_library(
        dataset,
        min_frequency=min_fragment_frequency,
        max_fragments=BRICS_MAX_FRAGMENTS,
    )

    if not library:
        # Fall back to frequency=1 if no fragments pass filter
        library = build_fragment_library(dataset, min_frequency=1)

    if not library:
        raise GenerationError("No BRICS fragments could be extracted from dataset")

    logger.info(f"Fragment library: {len(library)} fragments")

    # Enumerate new molecules
    products = enumerate_from_fragments(
        library,
        max_molecules=n_molecules,
        max_depth=max_depth,
        seed=seed,
    )

    # Get original SMILES for novelty checking
    original_smiles = {
        rec.canonical_smiles
        for rec in dataset.valid_records
        if rec.canonical_smiles
    }

    # Convert to MoleculeRecords
    records = []
    for i, mol in enumerate(products):
        smi = Chem.MolToSmiles(mol, canonical=True)
        is_novel = smi not in original_smiles
        rec = MoleculeRecord(
            mol=mol,
            source_id=f"brics_{i}",
            smiles=smi,
            status=MoleculeStatus.RAW,
        )
        rec.add_provenance("generated:brics")
        rec.metadata["generation_method"] = "brics"
        rec.metadata["is_novel"] = is_novel
        records.append(rec)

    result = MoleculeDataset(
        records=records,
        name=f"brics_generated_{len(records)}",
    )
    logger.info(f"BRICS generation complete: {len(records)} molecules")
    return result

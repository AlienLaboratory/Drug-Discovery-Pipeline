"""Tests for BRICS fragment enumeration."""

import pytest
from rdkit import Chem

from claudedd.core.exceptions import GenerationError
from claudedd.phase3.generation.brics_enum import (
    build_fragment_library,
    decompose_dataset,
    decompose_molecule,
    enumerate_from_fragments,
    generate_brics,
    recombine_two_molecules,
)


def test_decompose_molecule_aspirin(aspirin_mol):
    """Aspirin decomposes into BRICS fragments."""
    frags = decompose_molecule(aspirin_mol)
    assert isinstance(frags, list)
    assert len(frags) >= 1  # Should have at least one fragment


def test_decompose_molecule_none():
    """None molecule raises GenerationError."""
    with pytest.raises(GenerationError, match="Cannot decompose"):
        decompose_molecule(None)


def test_decompose_dataset(seed_dataset):
    """Dataset decomposition returns fragment dict."""
    result = decompose_dataset(seed_dataset)
    assert isinstance(result, dict)
    assert len(result) >= 1


def test_build_fragment_library(seed_dataset):
    """Fragment library built from dataset."""
    library = build_fragment_library(seed_dataset, min_frequency=1)
    assert isinstance(library, list)
    assert len(library) >= 1
    # All entries should be SMILES strings
    for frag in library:
        assert isinstance(frag, str)


def test_enumerate_from_fragments(seed_dataset):
    """Enumerate molecules from fragments."""
    library = build_fragment_library(seed_dataset, min_frequency=1)
    products = enumerate_from_fragments(library, max_molecules=5)
    assert isinstance(products, list)
    # Each product should be a valid mol
    for mol in products:
        assert mol is not None
        smi = Chem.MolToSmiles(mol)
        assert len(smi) > 0


def test_enumerate_from_empty_fragments():
    """Empty fragment list raises GenerationError."""
    with pytest.raises(GenerationError):
        enumerate_from_fragments([])


def test_recombine_two_molecules(aspirin_mol):
    """Recombine two molecules creates hybrids."""
    ibuprofen = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    products = recombine_two_molecules(aspirin_mol, ibuprofen, max_products=5)
    assert isinstance(products, list)
    # Products should be different from parents
    parent1_smi = Chem.MolToSmiles(aspirin_mol, canonical=True)
    parent2_smi = Chem.MolToSmiles(ibuprofen, canonical=True)
    for mol in products:
        smi = Chem.MolToSmiles(mol, canonical=True)
        assert smi != parent1_smi or smi != parent2_smi


def test_recombine_none_molecule(aspirin_mol):
    """Recombine with None raises GenerationError."""
    with pytest.raises(GenerationError):
        recombine_two_molecules(aspirin_mol, None)


def test_generate_brics(seed_dataset):
    """End-to-end BRICS generation."""
    generated = generate_brics(seed_dataset, n_molecules=10, seed=42)
    assert len(generated) > 0
    for rec in generated.valid_records:
        assert rec.mol is not None
        assert "brics" in rec.provenance[0]
        assert rec.metadata["generation_method"] == "brics"

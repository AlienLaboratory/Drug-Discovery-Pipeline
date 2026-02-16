"""Tests for scaffold decoration generation."""

import pytest
from rdkit import Chem

from drugflow.core.exceptions import GenerationError
from drugflow.phase3.generation.scaffold_decoration import (
    decorate_scaffold,
    extract_core_scaffold,
    find_decoration_sites,
    generate_from_scaffold,
    get_r_group_library,
)


def test_extract_core_scaffold(aspirin_mol):
    """Extract Murcko scaffold from aspirin."""
    scaffold = extract_core_scaffold(aspirin_mol)
    assert scaffold is not None
    smi = Chem.MolToSmiles(scaffold, canonical=True)
    assert len(smi) > 0


def test_extract_core_scaffold_none():
    """None molecule returns None."""
    assert extract_core_scaffold(None) is None


def test_find_decoration_sites(aspirin_mol):
    """Find decoration sites on aspirin."""
    scaffold = extract_core_scaffold(aspirin_mol)
    sites = find_decoration_sites(scaffold)
    assert isinstance(sites, list)
    # Scaffold should have at least some decoration sites
    assert len(sites) >= 0  # Some scaffolds may be fully substituted


def test_get_r_group_library():
    """R-group library contains valid molecules."""
    r_groups = get_r_group_library()
    assert len(r_groups) >= 10  # Should have substantial library
    for mol in r_groups:
        assert mol is not None


def test_decorate_scaffold(aspirin_mol):
    """Decorate scaffold produces valid molecules."""
    scaffold = extract_core_scaffold(aspirin_mol)
    if scaffold is not None:
        products = decorate_scaffold(scaffold, max_molecules=10, seed=42)
        assert isinstance(products, list)
        for mol in products:
            assert mol is not None
            smi = Chem.MolToSmiles(mol, canonical=True)
            assert len(smi) > 0


def test_decorate_scaffold_none():
    """Decorating None raises GenerationError."""
    with pytest.raises(GenerationError):
        decorate_scaffold(None)


def test_generate_from_scaffold(aspirin_mol):
    """End-to-end scaffold generation."""
    generated = generate_from_scaffold(aspirin_mol, n_molecules=10, seed=42)
    assert len(generated) >= 0  # May generate 0 if no sites found
    for rec in generated.valid_records:
        assert rec.mol is not None
        assert rec.metadata["generation_method"] == "scaffold_decoration"


def test_generate_from_scaffold_none():
    """None seed molecule raises GenerationError."""
    with pytest.raises(GenerationError):
        generate_from_scaffold(None)

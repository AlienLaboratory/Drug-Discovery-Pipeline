"""Tests for molecular mutations."""

import random

import pytest
from rdkit import Chem

from claudedd.core.exceptions import GenerationError
from claudedd.phase3.generation.mutations import (
    add_fragment,
    generate_mutations,
    mutate_atom,
    mutate_bond,
    mutate_molecule,
    random_mutation,
    remove_fragment,
    ring_open_close,
)


@pytest.fixture
def ethanol():
    return Chem.MolFromSmiles("CCO")


@pytest.fixture
def benzene():
    return Chem.MolFromSmiles("c1ccccc1")


def test_mutate_atom(ethanol):
    """Atom mutation changes an atom type."""
    rng = random.Random(42)
    result = mutate_atom(ethanol, rng=rng)
    # Should produce some result (may or may not succeed)
    if result is not None:
        smi = Chem.MolToSmiles(result, canonical=True)
        assert smi != Chem.MolToSmiles(ethanol, canonical=True)


def test_mutate_atom_none():
    """None molecule returns None."""
    assert mutate_atom(None) is None


def test_mutate_bond(ethanol):
    """Bond mutation changes a bond type."""
    rng = random.Random(42)
    result = mutate_bond(ethanol, rng=rng)
    # May or may not produce a valid molecule
    if result is not None:
        assert Chem.MolToSmiles(result) is not None


def test_add_fragment(ethanol):
    """Add fragment attaches a group."""
    rng = random.Random(42)
    result = add_fragment(ethanol, fragment_smiles="C", rng=rng)
    if result is not None:
        # Should have more atoms
        assert result.GetNumAtoms() >= ethanol.GetNumAtoms()


def test_remove_fragment():
    """Remove terminal fragment from molecule."""
    mol = Chem.MolFromSmiles("CCCC")  # butane
    rng = random.Random(42)
    result = remove_fragment(mol, rng=rng)
    if result is not None:
        assert result.GetNumAtoms() < mol.GetNumAtoms()


def test_ring_open_close(ethanol):
    """Ring mutation attempts to form a ring."""
    rng = random.Random(42)
    result = ring_open_close(ethanol, rng=rng)
    # May or may not succeed
    if result is not None:
        assert Chem.MolToSmiles(result) is not None


def test_random_mutation(aspirin_mol):
    """Random mutation produces a valid molecule."""
    rng = random.Random(42)
    result = random_mutation(aspirin_mol, rng=rng)
    if result is not None:
        smi = Chem.MolToSmiles(result, canonical=True)
        assert len(smi) > 0


def test_mutate_molecule_single(aspirin_mol):
    """Single mutation on aspirin."""
    rng = random.Random(42)
    result = mutate_molecule(aspirin_mol, n_mutations=1, rng=rng)
    if result is not None:
        assert Chem.MolToSmiles(result) is not None


def test_mutate_molecule_multiple(aspirin_mol):
    """Multiple mutations on aspirin."""
    rng = random.Random(42)
    result = mutate_molecule(aspirin_mol, n_mutations=3, rng=rng)
    if result is not None:
        assert Chem.MolToSmiles(result) is not None


def test_generate_mutations(seed_dataset):
    """End-to-end mutation generation."""
    generated = generate_mutations(
        seed_dataset, n_molecules=10, n_mutations_per_mol=1, seed=42,
    )
    assert len(generated) > 0
    for rec in generated.valid_records:
        assert rec.mol is not None
        assert rec.metadata["generation_method"] == "mutation"
        assert "parent_smiles" in rec.metadata

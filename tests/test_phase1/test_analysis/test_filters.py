"""Tests for drug-likeness filters."""

from rdkit import Chem

from drugflow.phase1.analysis.filters import (
    apply_lipinski,
    apply_pains,
    apply_veber,
    apply_all_filters,
)


def test_lipinski_aspirin_passes(aspirin_mol):
    result = apply_lipinski(aspirin_mol)
    assert result.passed
    assert len(result.violations) == 0


def test_lipinski_violations():
    # Large molecule with multiple violations
    big_mol = Chem.MolFromSmiles(
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"  # high MW, high LogP
    )
    result = apply_lipinski(big_mol, max_violations=0)
    assert not result.passed
    assert len(result.violations) > 0


def test_pains_benzoquinone():
    mol = Chem.MolFromSmiles("O=C1C=CC(=O)C=C1")
    result = apply_pains(mol)
    assert not result.passed


def test_pains_aspirin_passes(aspirin_mol):
    result = apply_pains(aspirin_mol)
    assert result.passed


def test_veber_aspirin_passes(aspirin_mol):
    result = apply_veber(aspirin_mol)
    assert result.passed


def test_apply_all_filters(aspirin_mol):
    results = apply_all_filters(aspirin_mol, lipinski=True, pains=True)
    assert "lipinski" in results
    assert "pains" in results
    assert results["lipinski"].passed
    assert results["pains"].passed

"""Tests for property computation."""

import pytest
from rdkit import Chem

from claudedd.phase1.analysis.properties import compute_properties


def test_compute_properties_aspirin(aspirin_mol):
    props = compute_properties(aspirin_mol)
    assert 170 < props.molecular_weight < 190  # ~180.16
    assert 0.5 < props.logp < 2.0
    assert props.hbd >= 0
    assert props.hba >= 0
    assert 0.0 <= props.qed <= 1.0


def test_compute_properties_caffeine():
    mol = Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
    props = compute_properties(mol)
    assert 190 < props.molecular_weight < 200  # ~194.19
    assert props.hbd == 0  # caffeine has no HBD


def test_qed_range(aspirin_mol):
    props = compute_properties(aspirin_mol)
    assert 0.0 <= props.qed <= 1.0

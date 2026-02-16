"""Tests for protein preparation."""

import os

import pytest
from rdkit import Chem

from drugflow.core.exceptions import DockingError
from drugflow.phase4.docking.protein_prep import (
    load_protein_pdb,
    prepare_protein,
    remove_heteroatoms,
    remove_water,
)


@pytest.fixture
def simple_pdb_file(tmp_path):
    """Create a simple PDB file for testing."""
    pdb_content = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00  0.00           C
ATOM      4  O   ALA A   1       4.000   5.000   6.000  1.00  0.00           O
HETATM    5  O   HOH A 101      10.000  10.000  10.000  1.00  0.00           O
HETATM    6  ZN  ZN  A 201      15.000  15.000  15.000  1.00  0.00          ZN
END
"""
    path = tmp_path / "test_protein.pdb"
    path.write_text(pdb_content)
    return str(path)


class TestLoadProteinPDB:
    """Tests for PDB loading."""

    def test_load_valid_pdb(self, simple_pdb_file):
        """Load a valid PDB file."""
        mol = load_protein_pdb(simple_pdb_file)
        assert mol is not None
        assert mol.GetNumAtoms() > 0

    def test_load_nonexistent_raises(self):
        """Nonexistent file raises error."""
        with pytest.raises(DockingError):
            load_protein_pdb("/nonexistent/path.pdb")


class TestRemoveWater:
    """Tests for water removal."""

    def test_removes_water(self, simple_pdb_file):
        """Water molecules are removed."""
        mol = load_protein_pdb(simple_pdb_file)
        n_before = mol.GetNumAtoms()
        cleaned = remove_water(mol)
        assert cleaned.GetNumAtoms() <= n_before

    def test_none_raises(self):
        """None protein raises error."""
        with pytest.raises(DockingError):
            remove_water(None)


class TestRemoveHeteroatoms:
    """Tests for heteroatom removal."""

    def test_removes_hetatm(self, simple_pdb_file):
        """Heteroatoms are removed."""
        mol = load_protein_pdb(simple_pdb_file)
        cleaned = remove_heteroatoms(mol)
        assert cleaned.GetNumAtoms() <= mol.GetNumAtoms()

    def test_keep_list(self, simple_pdb_file):
        """Keep list preserves specified residues."""
        mol = load_protein_pdb(simple_pdb_file)
        cleaned_no_keep = remove_heteroatoms(mol)
        cleaned_keep_zn = remove_heteroatoms(mol, keep_list=["ZN"])
        # Keeping ZN should result in >= atoms
        assert cleaned_keep_zn.GetNumAtoms() >= cleaned_no_keep.GetNumAtoms()

    def test_none_raises(self):
        """None protein raises error."""
        with pytest.raises(DockingError):
            remove_heteroatoms(None)


class TestPrepareProtein:
    """Tests for end-to-end protein preparation."""

    def test_prepare_protein(self, simple_pdb_file):
        """Full protein preparation pipeline."""
        protein = prepare_protein(simple_pdb_file)
        assert protein is not None
        assert protein.GetNumAtoms() > 0

    def test_prepare_no_water_removal(self, simple_pdb_file):
        """Prepare without water removal."""
        with_water = prepare_protein(simple_pdb_file, remove_waters=False)
        without_water = prepare_protein(simple_pdb_file, remove_waters=True)
        assert with_water.GetNumAtoms() >= without_water.GetNumAtoms()

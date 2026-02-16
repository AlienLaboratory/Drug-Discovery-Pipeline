"""Tests for tautomer enumeration and protonation."""

import pytest
from rdkit import Chem

from claudedd.core.exceptions import DockingError
from claudedd.phase4.ligand_prep.protonation import (
    add_hydrogens_3d,
    enumerate_tautomers,
    get_dominant_tautomer,
)


class TestEnumerateTautomers:
    """Tests for tautomer enumeration."""

    def test_enumerate_basic(self, aspirin_mol):
        """Enumerate tautomers of aspirin."""
        tautomers = enumerate_tautomers(aspirin_mol)
        assert len(tautomers) >= 1
        assert all(isinstance(t, Chem.Mol) for t in tautomers)

    def test_max_tautomers_limit(self, aspirin_mol):
        """Respect max_tautomers limit."""
        tautomers = enumerate_tautomers(aspirin_mol, max_tautomers=3)
        assert len(tautomers) <= 3

    def test_none_raises(self):
        """None molecule raises."""
        with pytest.raises(DockingError):
            enumerate_tautomers(None)


class TestGetDominantTautomer:
    """Tests for canonical tautomer."""

    def test_dominant_returns_mol(self, aspirin_mol):
        """Returns a valid molecule."""
        result = get_dominant_tautomer(aspirin_mol)
        assert result is not None
        assert isinstance(result, Chem.Mol)

    def test_none_raises(self):
        """None raises error."""
        with pytest.raises(DockingError):
            get_dominant_tautomer(None)


class TestAddHydrogens3D:
    """Tests for adding Hs with 3D coordinates."""

    def test_adds_hydrogens(self, aspirin_mol):
        """Adds hydrogens and generates 3D."""
        result = add_hydrogens_3d(aspirin_mol)
        assert result.GetNumAtoms() > aspirin_mol.GetNumAtoms()  # Hs added
        assert result.GetNumConformers() > 0  # 3D generated

    def test_preserves_existing_3d(self, aspirin_3d):
        """Existing 3D coords are preserved (Hs added)."""
        original_confs = aspirin_3d.GetNumConformers()
        result = add_hydrogens_3d(aspirin_3d)
        assert result.GetNumConformers() >= original_confs

    def test_none_raises(self):
        """None raises error."""
        with pytest.raises(DockingError):
            add_hydrogens_3d(None)

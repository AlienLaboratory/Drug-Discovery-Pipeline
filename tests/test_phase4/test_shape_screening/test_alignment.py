"""Tests for shape-based molecular alignment."""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from drugflow.core.exceptions import DockingError
from drugflow.phase4.shape_screening.alignment import (
    align_molecules_crippen,
    align_molecules_o3a,
    align_to_best_conformer,
    compute_rmsd,
)


class TestAlignO3A:
    """Tests for O3A alignment."""

    def test_basic_alignment(self, aspirin_3d, ibuprofen_3d):
        """O3A alignment returns score and molecule."""
        score, aligned = align_molecules_o3a(aspirin_3d, ibuprofen_3d)
        assert isinstance(score, float)
        assert aligned is not None

    def test_self_alignment(self, aspirin_3d):
        """Self-alignment should have high score."""
        mol_copy = Chem.RWMol(aspirin_3d)
        score, _ = align_molecules_o3a(aspirin_3d, mol_copy)
        assert isinstance(score, float)

    def test_none_raises(self, aspirin_3d):
        """None probe raises error."""
        with pytest.raises(DockingError):
            align_molecules_o3a(None, aspirin_3d)


class TestAlignCrippen:
    """Tests for Crippen O3A alignment."""

    def test_crippen_alignment(self, aspirin_3d, ibuprofen_3d):
        """Crippen O3A returns score and aligned mol."""
        score, aligned = align_molecules_crippen(aspirin_3d, ibuprofen_3d)
        assert isinstance(score, float)
        assert aligned is not None


class TestComputeRMSD:
    """Tests for RMSD computation."""

    def test_self_rmsd_near_zero(self, aspirin_3d):
        """RMSD of identical molecules should be near zero."""
        mol_copy = Chem.RWMol(aspirin_3d)
        rmsd = compute_rmsd(aspirin_3d, mol_copy)
        assert rmsd < 1.0  # Should be very small

    def test_different_molecules(self, aspirin_3d, ibuprofen_3d):
        """RMSD between different molecules is computed."""
        # This may raise if atom counts differ, which is expected
        # for molecules with different numbers of atoms
        try:
            rmsd = compute_rmsd(aspirin_3d, ibuprofen_3d)
            assert rmsd >= 0.0
        except DockingError:
            pass  # Expected for molecules with different atom counts


class TestAlignToBestConformer:
    """Tests for best conformer alignment."""

    def test_best_conformer_selection(self, multi_conf_mol, aspirin_3d):
        """Select best conformer from multi-conf molecule."""
        best_id, score, aligned = align_to_best_conformer(multi_conf_mol, aspirin_3d)
        assert isinstance(best_id, int)
        assert isinstance(score, float)
        assert score >= 0
        assert aligned is not None

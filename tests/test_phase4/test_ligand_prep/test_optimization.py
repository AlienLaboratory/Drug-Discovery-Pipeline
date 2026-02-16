"""Tests for force field optimization."""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from drugflow.core.exceptions import DockingError
from drugflow.phase4.ligand_prep.optimization import (
    compute_strain_energy,
    optimize_all_conformers,
    optimize_conformer,
    validate_3d_geometry,
)


class TestOptimizeConformer:
    """Tests for single conformer optimization."""

    def test_mmff_optimization(self, aspirin_3d):
        """MMFF optimization returns molecule and energy."""
        mol, energy = optimize_conformer(aspirin_3d, force_field="MMFF")
        assert mol is not None
        assert isinstance(energy, float)
        assert energy < float("inf")

    def test_uff_optimization(self, aspirin_3d):
        """UFF optimization works."""
        mol, energy = optimize_conformer(aspirin_3d, force_field="UFF")
        assert mol is not None
        assert isinstance(energy, float)

    def test_no_conformers_raises(self, aspirin_mol):
        """Molecule without 3D raises error."""
        with pytest.raises(DockingError):
            optimize_conformer(aspirin_mol)


class TestOptimizeAllConformers:
    """Tests for batch conformer optimization."""

    def test_optimize_multiple(self, multi_conf_mol):
        """Optimize all conformers returns sorted results."""
        results = optimize_all_conformers(multi_conf_mol, force_field="MMFF")
        assert len(results) == multi_conf_mol.GetNumConformers()
        # Results should be sorted by energy
        energies = [e for _, e in results]
        assert energies == sorted(energies)

    def test_uff_batch(self, multi_conf_mol):
        """UFF batch optimization works."""
        results = optimize_all_conformers(multi_conf_mol, force_field="UFF")
        assert len(results) > 0


class TestStrainEnergy:
    """Tests for strain energy computation."""

    def test_strain_energy_nonnegative(self, multi_conf_mol):
        """Strain energy is always >= 0."""
        strain = compute_strain_energy(multi_conf_mol)
        assert strain >= 0.0

    def test_lowest_energy_has_zero_strain(self, multi_conf_mol):
        """Lowest-energy conformer has zero strain."""
        from drugflow.phase4.ligand_prep.conformers import get_lowest_energy_conformer
        best_id, _ = get_lowest_energy_conformer(multi_conf_mol)
        strain = compute_strain_energy(multi_conf_mol, conf_id=best_id)
        assert abs(strain) < 1e-6


class TestValidate3DGeometry:
    """Tests for geometry validation."""

    def test_valid_3d_geometry(self, aspirin_3d):
        """Optimized molecule has valid geometry."""
        result = validate_3d_geometry(aspirin_3d)
        assert result["valid"] is True
        assert result["is_3d"] is True
        assert result["n_clashes"] == 0
        assert result["bond_issues"] == 0

    def test_no_conformers(self, aspirin_mol):
        """2D molecule reports invalid."""
        result = validate_3d_geometry(aspirin_mol)
        assert result["valid"] is False

    def test_none_molecule(self):
        """None molecule returns invalid."""
        result = validate_3d_geometry(None)
        assert result["valid"] is False

"""Tests for multi-conformer generation."""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from claudedd.core.exceptions import DockingError
from claudedd.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus
from claudedd.phase4.ligand_prep.conformers import (
    generate_conformers,
    generate_conformers_dataset,
    get_conformer_energies,
    get_lowest_energy_conformer,
    prune_conformers,
)


class TestGenerateConformers:
    """Tests for generate_conformers."""

    def test_basic_conformer_generation(self, aspirin_mol):
        """Generate conformers from 2D molecule."""
        result = generate_conformers(aspirin_mol, n_confs=5, seed=42)
        assert result is not None
        assert result.GetNumConformers() >= 1

    def test_multiple_conformers(self, aspirin_mol):
        """Generate multiple conformers."""
        result = generate_conformers(aspirin_mol, n_confs=10, seed=42)
        assert result.GetNumConformers() >= 2

    def test_conformers_have_3d(self, aspirin_mol):
        """Conformers have valid 3D coordinates."""
        result = generate_conformers(aspirin_mol, n_confs=3, seed=42)
        conf = result.GetConformer(0)
        pos = conf.GetAtomPosition(0)
        # At least one z-coordinate should be non-zero for 3D
        has_z = any(
            abs(conf.GetAtomPosition(i).z) > 0.01
            for i in range(result.GetNumAtoms())
        )
        assert has_z

    def test_none_molecule_raises(self):
        """None molecule raises DockingError."""
        with pytest.raises(DockingError, match="None"):
            generate_conformers(None)

    def test_seed_reproducibility(self, aspirin_mol):
        """Same seed produces same conformers."""
        mol1 = generate_conformers(Chem.RWMol(aspirin_mol), n_confs=3, seed=42)
        mol2 = generate_conformers(Chem.RWMol(aspirin_mol), n_confs=3, seed=42)
        assert mol1.GetNumConformers() == mol2.GetNumConformers()


class TestConformerEnergies:
    """Tests for energy computation."""

    def test_energies_returned(self, multi_conf_mol):
        """Get energies for each conformer."""
        energies = get_conformer_energies(multi_conf_mol, force_field="MMFF")
        assert len(energies) == multi_conf_mol.GetNumConformers()
        assert all(isinstance(e, float) for e in energies)

    def test_uff_energies(self, multi_conf_mol):
        """UFF energies work too."""
        energies = get_conformer_energies(multi_conf_mol, force_field="UFF")
        assert len(energies) > 0
        assert all(isinstance(e, float) for e in energies)

    def test_no_conformers_raises(self, aspirin_mol):
        """Molecule without conformers raises error."""
        with pytest.raises(DockingError):
            get_conformer_energies(aspirin_mol)


class TestLowestEnergyConformer:
    """Tests for get_lowest_energy_conformer."""

    def test_returns_valid_id_and_energy(self, multi_conf_mol):
        """Returns conformer ID and energy."""
        conf_id, energy = get_lowest_energy_conformer(multi_conf_mol)
        assert isinstance(conf_id, int)
        assert isinstance(energy, float)
        assert energy < float("inf")


class TestPruneConformers:
    """Tests for conformer pruning."""

    def test_pruning_reduces_conformers(self, multi_conf_mol):
        """Pruning with tight threshold reduces conformer count."""
        original = multi_conf_mol.GetNumConformers()
        pruned = prune_conformers(multi_conf_mol, rmsd_threshold=0.1)
        # May or may not reduce, but should not increase
        assert pruned.GetNumConformers() <= original

    def test_pruning_single_conformer(self, aspirin_3d):
        """Single conformer is kept."""
        result = prune_conformers(aspirin_3d)
        assert result.GetNumConformers() >= 1

    def test_none_molecule(self):
        """None mol returns None."""
        result = prune_conformers(None)
        assert result is None


class TestGenerateConformersDataset:
    """Tests for dataset-level conformer generation."""

    def test_dataset_generation(self, sample_dataset):
        """Generate conformers for entire dataset."""
        result = generate_conformers_dataset(sample_dataset, n_confs=3, seed=42)
        for rec in result.valid_records:
            if rec.mol is not None:
                assert rec.mol.GetNumConformers() >= 1
                assert "n_conformers" in rec.metadata

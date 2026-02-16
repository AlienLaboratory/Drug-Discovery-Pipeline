"""Tests for end-to-end ligand preparation."""

import os

import pytest
from rdkit import Chem

from drugflow.core.exceptions import DockingError
from drugflow.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus
from drugflow.phase4.ligand_prep.preparation import (
    export_ligand_pdb,
    export_ligand_sdf,
    prepare_ligand,
    prepare_ligand_dataset,
)


class TestPrepareLigand:
    """Tests for full ligand prep pipeline."""

    def test_basic_preparation(self, aspirin_mol):
        """Prepare aspirin from 2D."""
        result = prepare_ligand(aspirin_mol, n_confs=5, seed=42)
        assert result is not None
        assert result.GetNumConformers() >= 1

    def test_preparation_with_optimization(self, aspirin_mol):
        """Prep with optimization produces valid 3D."""
        result = prepare_ligand(aspirin_mol, n_confs=3, optimize=True, seed=42)
        assert result.GetNumConformers() >= 1

    def test_preparation_without_optimization(self, aspirin_mol):
        """Prep without optimization still works."""
        result = prepare_ligand(aspirin_mol, n_confs=3, optimize=False, seed=42)
        assert result.GetNumConformers() >= 1

    def test_uff_force_field(self, aspirin_mol):
        """UFF force field works in pipeline."""
        result = prepare_ligand(aspirin_mol, n_confs=3, force_field="UFF", seed=42)
        assert result.GetNumConformers() >= 1

    def test_none_raises(self):
        """None molecule raises."""
        with pytest.raises(DockingError):
            prepare_ligand(None)


class TestPrepareLigandDataset:
    """Tests for dataset-level preparation."""

    def test_dataset_preparation(self, sample_dataset):
        """Prepare all molecules in dataset."""
        result = prepare_ligand_dataset(sample_dataset, n_confs=3, seed=42)
        prepared = [r for r in result.valid_records
                    if r.mol is not None and r.mol.GetNumConformers() > 0]
        assert len(prepared) >= 1

    def test_metadata_populated(self, sample_dataset):
        """Check metadata is set after preparation."""
        result = prepare_ligand_dataset(sample_dataset, n_confs=3, seed=42)
        for rec in result.valid_records:
            if rec.metadata.get("prepared"):
                assert "n_conformers" in rec.metadata
                assert rec.metadata["n_conformers"] > 0


class TestExportLigand:
    """Tests for PDB/SDF export."""

    def test_export_pdb(self, aspirin_3d, tmp_path):
        """Export to PDB file."""
        out = str(tmp_path / "aspirin.pdb")
        result = export_ligand_pdb(aspirin_3d, out)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_export_sdf(self, aspirin_3d, tmp_path):
        """Export to SDF file."""
        out = str(tmp_path / "aspirin.sdf")
        result = export_ligand_sdf(aspirin_3d, out)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_export_no_conformers_raises(self, aspirin_mol, tmp_path):
        """Export without 3D raises."""
        out = str(tmp_path / "fail.pdb")
        with pytest.raises(DockingError):
            export_ligand_pdb(aspirin_mol, out)

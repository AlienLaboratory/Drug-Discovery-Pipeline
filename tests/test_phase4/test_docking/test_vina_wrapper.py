"""Tests for AutoDock Vina wrapper.

Most tests are skipped if Vina is not installed. Only the
availability check and error handling tests run unconditionally.
"""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from drugflow.core.exceptions import DockingError
from drugflow.phase4.docking.vina_wrapper import (
    _check_meeko_available,
    _check_vina_available,
    dock_vina,
)
from drugflow.phase4.docking.grid import DockingBox


class TestVinaAvailability:
    """Tests for Vina/Meeko availability detection."""

    def test_check_vina_returns_bool(self):
        """Vina check returns a boolean."""
        result = _check_vina_available()
        assert isinstance(result, bool)

    def test_check_meeko_returns_bool(self):
        """Meeko check returns a boolean."""
        result = _check_meeko_available()
        assert isinstance(result, bool)


class TestDockVina:
    """Tests for Vina docking (skip if not installed)."""

    def test_no_vina_raises_docking_error(self):
        """When Vina not installed, dock_vina raises DockingError."""
        if _check_vina_available():
            pytest.skip("Vina is installed — cannot test missing-vina path")

        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

        box = DockingBox(0, 0, 0, 20, 20, 20)
        with pytest.raises(DockingError, match="not installed"):
            dock_vina(mol, "dummy_protein.pdbqt", box)

    def test_none_ligand_raises(self):
        """None ligand raises error."""
        if not _check_vina_available():
            pytest.skip("Vina not installed")
        box = DockingBox(0, 0, 0, 20, 20, 20)
        with pytest.raises(DockingError):
            dock_vina(None, "protein.pdbqt", box)

    def test_missing_protein_raises(self):
        """Missing protein file raises error."""
        if not _check_vina_available():
            pytest.skip("Vina not installed")

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

        box = DockingBox(0, 0, 0, 20, 20, 20)
        with pytest.raises(DockingError):
            dock_vina(mol, "/nonexistent/protein.pdbqt", box)

"""Tests for docking box definition and validation."""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from drugflow.core.exceptions import DockingError
from drugflow.phase4.docking.grid import (
    DockingBox,
    define_grid_from_coords,
    define_grid_from_ligand,
    validate_grid,
)


class TestDockingBox:
    """Tests for DockingBox dataclass."""

    def test_creation(self):
        """Create a docking box."""
        box = DockingBox(
            center_x=10.0, center_y=20.0, center_z=30.0,
            size_x=20.0, size_y=20.0, size_z=20.0,
        )
        assert box.center == (10.0, 20.0, 30.0)
        assert box.size == (20.0, 20.0, 20.0)

    def test_volume(self):
        """Volume is product of sizes."""
        box = DockingBox(0, 0, 0, 10, 20, 30)
        assert box.volume == 6000.0

    def test_to_dict(self):
        """Converts to dict correctly."""
        box = DockingBox(1, 2, 3, 10, 20, 30)
        d = box.to_dict()
        assert d["center_x"] == 1
        assert d["size_z"] == 30


class TestDefineGridFromLigand:
    """Tests for grid from ligand."""

    def test_grid_from_3d_ligand(self, aspirin_3d):
        """Define grid from a 3D ligand."""
        box = define_grid_from_ligand(aspirin_3d)
        assert isinstance(box, DockingBox)
        assert box.size_x > 0
        assert box.size_y > 0
        assert box.size_z > 0

    def test_padding_increases_size(self, aspirin_3d):
        """More padding gives bigger box."""
        box5 = define_grid_from_ligand(aspirin_3d, padding=5.0)
        box15 = define_grid_from_ligand(aspirin_3d, padding=15.0)
        assert box15.size_x > box5.size_x
        assert box15.size_y > box5.size_y
        assert box15.size_z > box5.size_z

    def test_no_conformers_raises(self, aspirin_mol):
        """2D molecule raises error."""
        with pytest.raises(DockingError):
            define_grid_from_ligand(aspirin_mol)


class TestDefineGridFromCoords:
    """Tests for grid from coordinates."""

    def test_explicit_coords(self):
        """Create grid from explicit coordinates."""
        box = define_grid_from_coords(
            center=(10.0, 20.0, 30.0),
            size=(25.0, 25.0, 25.0),
        )
        assert box.center_x == 10.0
        assert box.size_x == 25.0


class TestValidateGrid:
    """Tests for grid validation."""

    def test_valid_grid(self):
        """Valid grid passes."""
        box = DockingBox(0, 0, 0, 20, 20, 20)
        assert validate_grid(box) is True

    def test_zero_size_invalid(self):
        """Zero-sized box is invalid."""
        box = DockingBox(0, 0, 0, 0, 20, 20)
        assert validate_grid(box) is False

    def test_too_small_invalid(self):
        """Very small box is invalid."""
        box = DockingBox(0, 0, 0, 2, 2, 2)
        assert validate_grid(box) is False

    def test_too_large_invalid(self):
        """Very large box is invalid."""
        box = DockingBox(0, 0, 0, 300, 20, 20)
        assert validate_grid(box) is False

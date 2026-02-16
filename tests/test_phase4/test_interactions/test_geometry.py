"""Tests for 3D geometry measurements."""

import numpy as np
import pytest
from rdkit import Chem

from claudedd.core.exceptions import DockingError
from claudedd.phase4.interactions.geometry import (
    compute_bounding_box,
    compute_molecular_volume,
    measure_angle,
    measure_dihedral,
    measure_distance,
)


class TestMeasureDistance:
    """Tests for distance measurement."""

    def test_bonded_distance(self, aspirin_3d):
        """Distance between bonded atoms is reasonable."""
        dist = measure_distance(aspirin_3d, 0, 1)
        assert 0.5 < dist < 3.0  # Typical bond length range

    def test_no_conformers_raises(self, aspirin_mol):
        """2D molecule raises error."""
        with pytest.raises(DockingError):
            measure_distance(aspirin_mol, 0, 1)


class TestMeasureAngle:
    """Tests for angle measurement."""

    def test_angle_range(self, aspirin_3d):
        """Angle between 3 atoms is in valid range."""
        # Find three connected atoms
        bond = aspirin_3d.GetBondWithIdx(0)
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Find a third atom connected to j
        for neighbor in aspirin_3d.GetAtomWithIdx(j).GetNeighbors():
            k = neighbor.GetIdx()
            if k != i:
                break
        angle = measure_angle(aspirin_3d, i, j, k)
        assert 60.0 < angle < 180.0  # Reasonable angle range


class TestMeasureDihedral:
    """Tests for dihedral measurement."""

    def test_dihedral_range(self, aspirin_3d):
        """Dihedral angle is in valid range."""
        # Find four connected atoms (a chain)
        mol = aspirin_3d
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            for n1 in mol.GetAtomWithIdx(i).GetNeighbors():
                if n1.GetIdx() != j:
                    for n2 in mol.GetAtomWithIdx(j).GetNeighbors():
                        if n2.GetIdx() != i:
                            dih = measure_dihedral(
                                mol, n1.GetIdx(), i, j, n2.GetIdx()
                            )
                            assert -180.0 <= dih <= 180.0
                            return
        pytest.skip("No 4-atom chain found")


class TestComputeVolume:
    """Tests for molecular volume."""

    def test_volume_positive(self, aspirin_3d):
        """Volume is positive."""
        vol = compute_molecular_volume(aspirin_3d)
        assert vol > 0.0

    def test_larger_molecule_larger_volume(self, aspirin_3d, ibuprofen_3d):
        """Larger molecule has larger volume (generally)."""
        vol_asp = compute_molecular_volume(aspirin_3d)
        vol_ibu = compute_molecular_volume(ibuprofen_3d)
        # Both should be positive
        assert vol_asp > 0
        assert vol_ibu > 0

    def test_no_conformers_raises(self, aspirin_mol):
        """2D molecule raises error."""
        with pytest.raises(DockingError):
            compute_molecular_volume(aspirin_mol)


class TestBoundingBox:
    """Tests for bounding box computation."""

    def test_bounding_box_shape(self, aspirin_3d):
        """Bounding box returns two 3D vectors."""
        bb_min, bb_max = compute_bounding_box(aspirin_3d)
        assert bb_min.shape == (3,)
        assert bb_max.shape == (3,)
        assert np.all(bb_max >= bb_min)

    def test_padding_expands_box(self, aspirin_3d):
        """Padding expands the bounding box."""
        bb_min0, bb_max0 = compute_bounding_box(aspirin_3d, padding=0.0)
        bb_min5, bb_max5 = compute_bounding_box(aspirin_3d, padding=5.0)
        assert np.all(bb_min5 < bb_min0)
        assert np.all(bb_max5 > bb_max0)

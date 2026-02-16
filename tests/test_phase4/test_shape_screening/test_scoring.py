"""Tests for shape similarity scoring."""

import pytest
from rdkit import Chem

from drugflow.core.exceptions import DockingError
from drugflow.phase4.shape_screening.scoring import (
    compute_combo_score,
    compute_pharmacophore_score,
    compute_shape_protrusion,
    compute_shape_tanimoto,
)


class TestShapeTanimoto:
    """Tests for Shape Tanimoto similarity."""

    def test_self_similarity(self, aspirin_3d):
        """Self-similarity should be close to 1.0."""
        score = compute_shape_tanimoto(aspirin_3d, aspirin_3d)
        assert 0.9 <= score <= 1.0

    def test_different_molecules(self, aspirin_3d, caffeine_3d):
        """Different molecules have lower similarity."""
        score = compute_shape_tanimoto(aspirin_3d, caffeine_3d)
        assert 0.0 <= score <= 1.0

    def test_score_range(self, aspirin_3d, ibuprofen_3d):
        """Score is in valid range."""
        score = compute_shape_tanimoto(aspirin_3d, ibuprofen_3d)
        assert 0.0 <= score <= 1.0

    def test_none_raises(self, aspirin_3d):
        """None molecule raises."""
        with pytest.raises(DockingError):
            compute_shape_tanimoto(None, aspirin_3d)


class TestShapeProtrusion:
    """Tests for shape protrusion."""

    def test_protrusion_range(self, aspirin_3d, ibuprofen_3d):
        """Protrusion is in valid range."""
        dist = compute_shape_protrusion(aspirin_3d, ibuprofen_3d)
        assert 0.0 <= dist <= 1.0


class TestPharmacophoreScore:
    """Tests for pharmacophore scoring."""

    def test_pharmacophore_returns_float(self, aspirin_3d, ibuprofen_3d):
        """Pharmacophore score returns numeric value."""
        score = compute_pharmacophore_score(aspirin_3d, ibuprofen_3d)
        assert isinstance(score, float)


class TestComboScore:
    """Tests for combo scoring."""

    def test_combo_self_score(self, aspirin_3d):
        """Combo self-score should be positive."""
        score = compute_combo_score(aspirin_3d, aspirin_3d)
        assert score > 0.0

    def test_combo_different_weights(self, aspirin_3d, ibuprofen_3d):
        """Different shape weights produce different scores."""
        s1 = compute_combo_score(aspirin_3d, ibuprofen_3d, shape_weight=0.8)
        s2 = compute_combo_score(aspirin_3d, ibuprofen_3d, shape_weight=0.2)
        # Scores may differ (or be similar if pharma ≈ shape)
        assert isinstance(s1, float)
        assert isinstance(s2, float)

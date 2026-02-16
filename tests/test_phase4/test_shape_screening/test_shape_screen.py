"""Tests for shape-based virtual screening."""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from claudedd.core.exceptions import DockingError
from claudedd.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus
from claudedd.phase4.shape_screening.shape_screen import rank_by_shape, screen_by_shape


class TestScreenByShape:
    """Tests for shape screening."""

    def test_screen_returns_dataset(self, dataset_3d):
        """Screening returns a MoleculeDataset."""
        ref = dataset_3d.valid_records[0].mol
        hits = screen_by_shape(dataset_3d, ref, threshold=0.0)
        assert isinstance(hits, MoleculeDataset)

    def test_low_threshold_includes_all(self, dataset_3d):
        """Very low threshold includes most/all molecules."""
        ref = dataset_3d.valid_records[0].mol
        hits = screen_by_shape(dataset_3d, ref, threshold=0.0)
        assert len(hits.valid_records) >= 1

    def test_high_threshold_filters(self, dataset_3d):
        """High threshold reduces hit count."""
        ref = dataset_3d.valid_records[0].mol
        hits_low = screen_by_shape(dataset_3d, ref, threshold=0.0)
        hits_high = screen_by_shape(dataset_3d, ref, threshold=0.99)
        assert len(hits_high.valid_records) <= len(hits_low.valid_records)

    def test_none_reference_raises(self, dataset_3d):
        """None reference raises error."""
        with pytest.raises(DockingError):
            screen_by_shape(dataset_3d, None)

    def test_combo_metric(self, dataset_3d):
        """Combo metric works in screening."""
        ref = dataset_3d.valid_records[0].mol
        hits = screen_by_shape(dataset_3d, ref, threshold=0.0, metric="combo")
        assert isinstance(hits, MoleculeDataset)


class TestRankByShape:
    """Tests for shape ranking."""

    def test_rank_returns_sorted(self, dataset_3d):
        """Ranking returns sorted results."""
        ref = dataset_3d.valid_records[0].mol
        ranked = rank_by_shape(dataset_3d, ref)
        assert len(ranked) >= 1
        # Check sorted descending
        scores = [s for _, _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_tuple_format(self, dataset_3d):
        """Each result is (index, record, score)."""
        ref = dataset_3d.valid_records[0].mol
        ranked = rank_by_shape(dataset_3d, ref)
        for idx, rec, score in ranked:
            assert isinstance(idx, int)
            assert isinstance(rec, MoleculeRecord)
            assert isinstance(score, float)

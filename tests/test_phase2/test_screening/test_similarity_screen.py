"""Tests for similarity-based screening."""

import numpy as np
import pytest

from drugflow.core.exceptions import ScreeningError
from drugflow.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus
from drugflow.phase2.screening.similarity_screen import (
    compute_max_similarity,
    compute_mean_similarity,
    extract_reference_fps,
    screen_similarity,
)


def test_compute_max_similarity():
    """Max similarity with identical fingerprint is 1.0."""
    fp = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    rec = MoleculeRecord(mol=None, fingerprints={"test_fp": fp})
    ref_fps = [fp]
    assert compute_max_similarity(rec, ref_fps, fp_type="test_fp") == 1.0


def test_compute_mean_similarity():
    """Mean similarity with references."""
    fp = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    ref1 = np.array([1, 0, 1, 1, 0], dtype=np.uint8)  # identical
    ref2 = np.array([0, 1, 0, 0, 1], dtype=np.uint8)  # different
    rec = MoleculeRecord(mol=None, fingerprints={"test_fp": fp})
    mean_sim = compute_mean_similarity(rec, [ref1, ref2], fp_type="test_fp")
    assert 0.0 < mean_sim < 1.0


def test_extract_reference_fps(computed_dataset):
    """Extract fingerprints from dataset."""
    fps = extract_reference_fps(computed_dataset, fp_type="morgan_r2_2048")
    assert len(fps) == len(computed_dataset.valid_records)
    assert all(isinstance(fp, np.ndarray) for fp in fps)


def test_extract_reference_fps_missing():
    """Missing fingerprints raises ScreeningError."""
    ds = MoleculeDataset(records=[], name="empty")
    with pytest.raises(ScreeningError, match="No fingerprints"):
        extract_reference_fps(ds, fp_type="nonexistent")


def test_screen_similarity(computed_dataset):
    """Similarity screen with self-reference should find all molecules."""
    hits = screen_similarity(
        computed_dataset, computed_dataset,
        threshold=0.5, fp_type="morgan_r2_2048",
    )
    # All molecules should be similar to themselves
    assert len(hits) > 0


def test_screen_similarity_high_threshold(computed_dataset):
    """Very high threshold filters most molecules."""
    hits = screen_similarity(
        computed_dataset, computed_dataset,
        threshold=0.99, fp_type="morgan_r2_2048",
    )
    # Self-similarity = 1.0, so still should pass
    assert len(hits) > 0


def test_screen_similarity_invalid_threshold(computed_dataset):
    """Invalid threshold raises ScreeningError."""
    with pytest.raises(ScreeningError, match="Threshold must be 0-1"):
        screen_similarity(
            computed_dataset, computed_dataset, threshold=1.5,
        )


def test_screen_similarity_properties_stored(computed_dataset):
    """Screening stores sim_screen_max in properties."""
    screen_similarity(
        computed_dataset, computed_dataset,
        threshold=0.0, fp_type="morgan_r2_2048",
    )
    for rec in computed_dataset.valid_records:
        assert "sim_screen_max" in rec.properties
        assert "sim_screen_pass" in rec.properties

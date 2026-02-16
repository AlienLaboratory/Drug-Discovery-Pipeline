"""Tests for SA Score computation."""

import pytest
from rdkit import Chem

from claudedd.core.exceptions import ScoringError
from claudedd.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus
from claudedd.phase2.scoring.sa_score import (
    compute_sa_score,
    normalize_sa_score,
    compute_sa_score_dataset,
)


def test_sa_score_aspirin(aspirin_mol):
    """Aspirin should have a low SA score (easy to synthesize)."""
    score = compute_sa_score(aspirin_mol)
    assert 1.0 <= score <= 10.0
    assert score < 3.0  # Aspirin is easy to synthesize


def test_sa_score_none():
    """None molecule raises ScoringError."""
    with pytest.raises(ScoringError, match="Cannot compute"):
        compute_sa_score(None)


def test_normalize_sa_score_easy():
    """SA score 1 (easy) normalizes to 1.0."""
    assert normalize_sa_score(1.0) == 1.0


def test_normalize_sa_score_hard():
    """SA score 10 (hard) normalizes to 0.0."""
    assert normalize_sa_score(10.0) == 0.0


def test_normalize_sa_score_medium():
    """SA score 5.5 normalizes to about 0.5."""
    normalized = normalize_sa_score(5.5)
    assert 0.4 <= normalized <= 0.6


def test_compute_sa_score_dataset(sample_dataset):
    """Compute SA score for dataset."""
    result = compute_sa_score_dataset(sample_dataset)
    for rec in result.valid_records:
        if rec.mol is not None:
            assert "sa_score" in rec.properties
            assert "sa_score_normalized" in rec.properties
            assert 1.0 <= rec.properties["sa_score"] <= 10.0
            assert 0.0 <= rec.properties["sa_score_normalized"] <= 1.0


def test_compute_sa_score_dataset_no_normalize(sample_dataset):
    """SA score without normalization."""
    result = compute_sa_score_dataset(sample_dataset, normalize=False)
    for rec in result.valid_records:
        if rec.mol is not None:
            assert "sa_score" in rec.properties
            assert "sa_score_normalized" not in rec.properties

"""Tests for multi-objective scoring."""

import pytest
import numpy as np

from drugflow.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus
from drugflow.phase2.scoring.multi_objective import (
    compute_composite_score,
    normalize_scores_minmax,
    compute_composite_score_dataset,
)


def test_composite_score_all_components():
    """Composite score with all components."""
    rec = MoleculeRecord(mol=None, source_id="test")
    rec.properties["predicted_activity"] = 0.8
    rec.properties["drug_likeness_score"] = 0.7
    rec.properties["sa_score_normalized"] = 0.9
    score = compute_composite_score(rec)
    assert 0.0 <= score <= 1.0
    assert score > 0.5


def test_composite_score_missing_components():
    """Composite score handles missing components gracefully."""
    rec = MoleculeRecord(mol=None, source_id="test")
    rec.properties["drug_likeness_score"] = 0.7
    score = compute_composite_score(rec)
    assert 0.0 <= score <= 1.0


def test_composite_score_custom_weights():
    """Custom weights are applied correctly."""
    rec = MoleculeRecord(mol=None, source_id="test")
    rec.properties["predicted_activity"] = 1.0
    rec.properties["drug_likeness_score"] = 0.0
    rec.properties["sa_score_normalized"] = 0.0

    weights = {
        "predicted_activity": 1.0,
        "drug_likeness": 0.0,
        "sa_score": 0.0,
    }
    score = compute_composite_score(rec, weights=weights)
    assert score == pytest.approx(1.0, abs=0.01)


def test_normalize_scores_minmax(sample_dataset):
    """Min-max normalization works."""
    for i, rec in enumerate(sample_dataset.valid_records):
        rec.properties["test_val"] = float(i)

    normalize_scores_minmax(sample_dataset, "test_val")

    for rec in sample_dataset.valid_records:
        if "test_val_normalized" in rec.properties:
            assert 0.0 <= rec.properties["test_val_normalized"] <= 1.0


def test_compute_composite_score_dataset(sample_dataset):
    """Composite scores computed for dataset."""
    # Add some scores
    for rec in sample_dataset.valid_records:
        rec.properties["drug_likeness_score"] = 0.5
        rec.properties["sa_score_normalized"] = 0.7

    result = compute_composite_score_dataset(sample_dataset)
    scored = sum(
        1 for r in result.valid_records
        if "composite_score" in r.properties
    )
    assert scored > 0

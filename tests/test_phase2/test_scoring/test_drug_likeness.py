"""Tests for drug-likeness scoring."""

import pytest
from rdkit import Chem

from claudedd.core.models import MoleculeRecord, MoleculeStatus
from claudedd.phase2.scoring.drug_likeness import (
    compute_drug_likeness,
    compute_drug_likeness_dataset,
)


def test_drug_likeness_with_qed(aspirin_record):
    """Drug-likeness uses QED when available."""
    aspirin_record.properties["QED"] = 0.55
    aspirin_record.properties["lipinski_pass"] = True
    aspirin_record.properties["pains_pass"] = True
    score = compute_drug_likeness(aspirin_record)
    assert 0.0 <= score <= 1.0
    assert score > 0.3  # Should be moderately drug-like


def test_drug_likeness_all_pass(aspirin_record):
    """Molecule passing all filters gets high score."""
    aspirin_record.properties["QED"] = 0.9
    aspirin_record.properties["lipinski_pass"] = True
    aspirin_record.properties["pains_pass"] = True
    aspirin_record.properties["veber_pass"] = True
    score = compute_drug_likeness(aspirin_record)
    assert score > 0.7


def test_drug_likeness_all_fail(aspirin_record):
    """Molecule failing all filters gets low score."""
    aspirin_record.properties["QED"] = 0.1
    aspirin_record.properties["lipinski_pass"] = False
    aspirin_record.properties["pains_pass"] = False
    aspirin_record.properties["veber_pass"] = False
    score = compute_drug_likeness(aspirin_record)
    assert score < 0.3


def test_drug_likeness_no_filters(aspirin_record):
    """Without filters, still computes from QED."""
    score = compute_drug_likeness(aspirin_record)
    assert 0.0 <= score <= 1.0


def test_drug_likeness_dataset(sample_dataset):
    """Compute drug-likeness for entire dataset."""
    result = compute_drug_likeness_dataset(sample_dataset)
    scored_count = sum(
        1 for r in result.valid_records
        if "drug_likeness_score" in r.properties
    )
    assert scored_count > 0

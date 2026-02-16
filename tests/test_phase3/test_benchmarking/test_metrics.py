"""Tests for benchmarking metrics."""

import pytest
from rdkit import Chem

from claudedd.phase3.benchmarking.metrics import (
    compute_all_metrics,
    compute_drug_likeness_rate,
    compute_internal_diversity,
    compute_novelty_rate,
    compute_qed_distribution,
    compute_sa_score_distribution,
    compute_uniqueness_rate,
    compute_validity_rate,
)


@pytest.fixture
def valid_mols():
    """List of valid RDKit molecules."""
    smiles = ["CCO", "CCCO", "c1ccccc1", "CC(=O)O", "CCN"]
    return [Chem.MolFromSmiles(s) for s in smiles]


@pytest.fixture
def mixed_mols():
    """List including None (invalid) molecules."""
    return [
        Chem.MolFromSmiles("CCO"),
        None,
        Chem.MolFromSmiles("c1ccccc1"),
        None,
        Chem.MolFromSmiles("CCN"),
    ]


def test_validity_rate_all_valid(valid_mols):
    """All valid molecules give 1.0 validity."""
    assert compute_validity_rate(valid_mols) == 1.0


def test_validity_rate_mixed(mixed_mols):
    """Mixed valid/invalid gives correct rate."""
    rate = compute_validity_rate(mixed_mols)
    assert rate == pytest.approx(0.6, abs=0.01)


def test_validity_rate_empty():
    """Empty list gives 0.0."""
    assert compute_validity_rate([]) == 0.0


def test_uniqueness_rate(valid_mols):
    """All unique molecules give 1.0 uniqueness."""
    assert compute_uniqueness_rate(valid_mols) == 1.0


def test_uniqueness_rate_duplicates():
    """Duplicate molecules reduce uniqueness."""
    mols = [Chem.MolFromSmiles("CCO")] * 5
    rate = compute_uniqueness_rate(mols)
    assert rate == pytest.approx(0.2, abs=0.01)


def test_novelty_rate(valid_mols):
    """Novelty vs reference set."""
    ref = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCCO")]
    rate = compute_novelty_rate(valid_mols, ref)
    assert 0.0 <= rate <= 1.0
    assert rate > 0.0  # Some should be novel


def test_internal_diversity(valid_mols):
    """Internal diversity between 0 and 1."""
    div = compute_internal_diversity(valid_mols)
    assert 0.0 <= div <= 1.0


def test_drug_likeness_rate(valid_mols):
    """Drug-likeness rate for small molecules (all pass Lipinski)."""
    rate = compute_drug_likeness_rate(valid_mols)
    assert rate == 1.0  # Small molecules pass Lipinski


def test_sa_score_distribution(valid_mols):
    """SA score distribution has expected keys."""
    dist = compute_sa_score_distribution(valid_mols)
    assert "mean_sa_score" in dist
    assert "median_sa_score" in dist
    assert "std_sa_score" in dist
    assert dist["mean_sa_score"] > 0


def test_qed_distribution(valid_mols):
    """QED distribution has expected keys."""
    dist = compute_qed_distribution(valid_mols)
    assert "mean_qed" in dist
    assert 0.0 <= dist["mean_qed"] <= 1.0


def test_compute_all_metrics(valid_mols):
    """All metrics computed at once."""
    ref = [Chem.MolFromSmiles("CCO")]
    metrics = compute_all_metrics(valid_mols, ref)
    assert "validity" in metrics
    assert "uniqueness" in metrics
    assert "novelty" in metrics
    assert "internal_diversity" in metrics
    assert "drug_likeness_rate" in metrics
    assert "mean_sa_score" in metrics
    assert "mean_qed" in metrics

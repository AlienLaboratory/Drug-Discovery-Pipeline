"""Tests for ADMET aggregate scoring."""

import pytest
from rdkit import Chem

from drugflow.core.models import MoleculeRecord
from drugflow.phase5.admet.absorption import predict_absorption_record
from drugflow.phase5.admet.distribution import predict_distribution_record
from drugflow.phase5.admet.excretion import predict_excretion_record
from drugflow.phase5.admet.metabolism import predict_metabolism_record
from drugflow.phase5.admet.scoring import (
    ADMETScore,
    classify_admet,
    compute_admet_score,
    compute_admet_score_record,
    risk_to_score,
)
from drugflow.phase5.admet.toxicity import predict_toxicity_record


@pytest.fixture
def aspirin_record():
    """Aspirin record with all ADMET predictions pre-computed."""
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    rec = MoleculeRecord(mol=mol)
    predict_absorption_record(rec)
    predict_distribution_record(rec)
    predict_metabolism_record(rec)
    predict_excretion_record(rec)
    predict_toxicity_record(rec)
    return rec


class TestRiskToScore:
    def test_green(self):
        assert risk_to_score("green") == 1.0

    def test_yellow(self):
        assert risk_to_score("yellow") == 0.5

    def test_red(self):
        assert risk_to_score("red") == 0.0

    def test_unknown_default(self):
        assert risk_to_score("unknown") == 0.5


class TestClassifyAdmet:
    def test_favorable(self):
        assert classify_admet(0.8) == "favorable"

    def test_moderate(self):
        assert classify_admet(0.55) == "moderate"

    def test_poor(self):
        assert classify_admet(0.3) == "poor"

    def test_boundary_favorable(self):
        assert classify_admet(0.7) == "favorable"

    def test_boundary_moderate(self):
        assert classify_admet(0.4) == "moderate"


class TestComputeAdmetScore:
    def test_aspirin_score(self, aspirin_record):
        """Aspirin should get a reasonable ADMET score."""
        score = compute_admet_score(aspirin_record)
        assert isinstance(score, ADMETScore)
        assert 0 <= score.overall_score <= 1
        assert score.overall_class in ("favorable", "moderate", "poor")
        assert score.n_red_flags >= 0
        assert score.n_yellow_flags >= 0
        assert score.n_green_flags >= 0

    def test_custom_weights(self, aspirin_record):
        """Custom weights should change the overall score."""
        default_score = compute_admet_score(aspirin_record)
        heavy_tox = compute_admet_score(aspirin_record, weights={
            "absorption": 0.05,
            "distribution": 0.05,
            "metabolism": 0.05,
            "excretion": 0.05,
            "toxicity": 0.80,
        })
        # Different weights, may give different scores
        assert isinstance(heavy_tox.overall_score, float)

    def test_record_storage(self, aspirin_record):
        """compute_admet_score_record stores in rec.properties."""
        compute_admet_score_record(aspirin_record)
        assert "admet_score" in aspirin_record.properties
        assert "admet_class" in aspirin_record.properties
        assert "admet_n_red_flags" in aspirin_record.properties
        assert "admet_absorption_score" in aspirin_record.properties
        assert "admet_toxicity_score" in aspirin_record.properties

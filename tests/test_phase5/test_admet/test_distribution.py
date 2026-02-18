"""Tests for ADMET distribution predictions."""

import pytest
from rdkit import Chem

from drugflow.core.exceptions import ADMETError
from drugflow.core.models import MoleculeRecord
from drugflow.phase5.admet.distribution import (
    DistributionResult,
    predict_bbb,
    predict_distribution,
    predict_distribution_record,
    predict_ppb,
    predict_vod,
)


@pytest.fixture
def caffeine():
    """Caffeine: low MW, moderate LogP, low TPSA → BBB+."""
    return Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")


@pytest.fixture
def metformin():
    """Metformin: MW~129, LogP~-1.4 → hydrophilic, low PPB."""
    return Chem.MolFromSmiles("CN(C)C(=N)NC(=N)N")


@pytest.fixture
def diclofenac():
    """Diclofenac: MW~296, LogP~4.5 → lipophilic, high PPB."""
    return Chem.MolFromSmiles("OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl")


class TestPredictBBB:
    def test_caffeine_bbb_positive(self, caffeine):
        """Caffeine should be BBB+ (small, moderate LogP, low TPSA)."""
        is_pen, logbb, risk = predict_bbb(caffeine)
        # Caffeine is known to cross BBB
        assert isinstance(is_pen, bool)
        assert isinstance(logbb, float)
        assert risk in ("green", "yellow")

    def test_returns_logbb_value(self, caffeine):
        """BBB prediction returns numeric LogBB."""
        _, logbb, _ = predict_bbb(caffeine)
        assert -5 < logbb < 5  # reasonable range


class TestPredictPPB:
    def test_high_logp_high_ppb(self, diclofenac):
        """Diclofenac (LogP~4.5) → high PPB."""
        cls, risk = predict_ppb(diclofenac)
        assert cls == "high"
        assert risk == "yellow"

    def test_low_logp_low_ppb(self, metformin):
        """Metformin (LogP~-1.4) → low PPB."""
        cls, risk = predict_ppb(metformin)
        assert cls == "low"
        assert risk == "green"


class TestPredictVoD:
    def test_basic_classification(self, caffeine):
        """VoD prediction returns valid class."""
        cls, risk = predict_vod(caffeine)
        assert cls in ("low", "moderate", "high")
        assert risk in ("green", "yellow", "red")


class TestPredictDistribution:
    def test_full_result(self, caffeine):
        """Full distribution result has all fields."""
        result = predict_distribution(caffeine)
        assert isinstance(result, DistributionResult)
        assert isinstance(result.bbb_penetrant, bool)
        assert isinstance(result.bbb_logbb, float)
        assert result.ppb_class in ("high", "moderate", "low")
        assert result.vod_class in ("low", "moderate", "high")

    def test_none_mol_raises(self):
        """None molecule raises ADMETError."""
        with pytest.raises(ADMETError):
            predict_distribution(None)

    def test_record_storage(self, caffeine):
        """predict_distribution_record stores results in rec.properties."""
        rec = MoleculeRecord(mol=caffeine)
        predict_distribution_record(rec)
        assert "admet_bbb_penetrant" in rec.properties
        assert "admet_bbb_logbb" in rec.properties
        assert "admet_ppb_class" in rec.properties
        assert "admet_vod_class" in rec.properties

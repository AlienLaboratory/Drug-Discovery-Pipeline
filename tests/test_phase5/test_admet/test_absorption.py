"""Tests for ADMET absorption predictions."""

import pytest
from rdkit import Chem

from drugflow.core.exceptions import ADMETError
from drugflow.core.models import MoleculeRecord
from drugflow.phase5.admet.absorption import (
    AbsorptionResult,
    predict_absorption,
    predict_absorption_record,
    predict_caco2,
    predict_hia,
    predict_oral_bioavailability,
    predict_pgp_substrate,
)


@pytest.fixture
def aspirin():
    """Aspirin: MW~180, TPSA~63, LogP~1.2 — small, drug-like."""
    return Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")


@pytest.fixture
def large_polar():
    """Large polar molecule: high MW, high TPSA."""
    # Erythromycin-like: MW~734, TPSA~193
    return Chem.MolFromSmiles("CC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC2OC(C)CC(N(C)C)C2O)C(C)(O)CC(C)C(=O)C(C)C(O)C1(C)O")


@pytest.fixture
def caffeine():
    """Caffeine: MW~194, TPSA~58, LogP~-0.07."""
    return Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")


class TestPredictCaco2:
    def test_high_permeability(self, aspirin):
        """Aspirin has low TPSA (~63) → high permeability."""
        cls, risk = predict_caco2(aspirin)
        assert cls == "high"
        assert risk == "green"

    def test_low_permeability(self, large_polar):
        """Large polar molecule has high TPSA → low permeability."""
        cls, risk = predict_caco2(large_polar)
        assert cls == "low"
        assert risk == "red"


class TestPredictHIA:
    def test_drug_like_high_hia(self, aspirin):
        """Aspirin is small and drug-like → high HIA."""
        cls, risk = predict_hia(aspirin)
        assert cls == "high"
        assert risk == "green"

    def test_large_molecule_low_hia(self, large_polar):
        """Large polar molecule → poor HIA."""
        cls, risk = predict_hia(large_polar)
        assert cls == "low"
        assert risk == "red"


class TestPredictPgp:
    def test_small_molecule_not_substrate(self, aspirin):
        """Aspirin (MW~180) should not be Pgp substrate."""
        is_sub, risk = predict_pgp_substrate(aspirin)
        assert is_sub is False
        assert risk == "green"

    def test_large_polar_substrate(self, large_polar):
        """Large polar molecule → likely Pgp substrate."""
        is_sub, risk = predict_pgp_substrate(large_polar)
        assert is_sub is True
        assert risk == "yellow"


class TestPredictBioavailability:
    def test_aspirin_high_bioavailability(self, aspirin):
        """Aspirin should have high oral bioavailability score."""
        score, risk = predict_oral_bioavailability(aspirin)
        assert score >= 0.7
        assert risk == "green"


class TestPredictAbsorption:
    def test_full_result(self, aspirin):
        """Full absorption result has all fields."""
        result = predict_absorption(aspirin)
        assert isinstance(result, AbsorptionResult)
        assert result.caco2_class in ("high", "moderate", "low")
        assert result.hia_class in ("high", "moderate", "low")
        assert isinstance(result.pgp_substrate, bool)
        assert 0 <= result.bioavailability_score <= 1

    def test_none_mol_raises(self):
        """None molecule raises ADMETError."""
        with pytest.raises(ADMETError):
            predict_absorption(None)

    def test_record_storage(self, aspirin):
        """predict_absorption_record stores results in rec.properties."""
        rec = MoleculeRecord(mol=aspirin)
        predict_absorption_record(rec)
        assert "admet_caco2_class" in rec.properties
        assert "admet_hia_class" in rec.properties
        assert "admet_pgp_substrate" in rec.properties
        assert "admet_bioavailability_score" in rec.properties

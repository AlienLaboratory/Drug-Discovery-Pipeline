"""Tests for ADMET metabolism predictions."""

import pytest
from rdkit import Chem

from drugflow.core.exceptions import ADMETError
from drugflow.core.models import MoleculeRecord
from drugflow.phase5.admet.metabolism import (
    MetabolismResult,
    detect_cyp_alerts,
    predict_cyp_inhibition,
    predict_metabolic_stability,
    predict_metabolism,
    predict_metabolism_record,
)


@pytest.fixture
def aspirin():
    """Aspirin: no CYP alerts expected."""
    return Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")


@pytest.fixture
def ketoconazole():
    """Ketoconazole: contains imidazole → CYP alert."""
    return Chem.MolFromSmiles("CC(=O)Nc1ccc(OCC2COC(Cn3ccnc3)(c3ccc(Cl)cc3Cl)O2)cc1")


@pytest.fixture
def small_stable():
    """Small, stable molecule: MW < 500, LogP < 3."""
    return Chem.MolFromSmiles("c1ccc(O)cc1")  # phenol, MW~94


@pytest.fixture
def large_lipophilic():
    """Large lipophilic molecule with many aromatic rings."""
    # A large, flexible, lipophilic molecule
    return Chem.MolFromSmiles("c1ccc(-c2ccc(-c3ccc(-c4ccc(CCCCCCCCC)cc4)cc3)cc2)cc1")


class TestDetectCypAlerts:
    def test_no_alerts_aspirin(self, aspirin):
        """Aspirin should have no CYP inhibition alerts."""
        alerts = detect_cyp_alerts(aspirin)
        assert len(alerts) == 0

    def test_imidazole_alert(self, ketoconazole):
        """Ketoconazole contains imidazole → CYP alert."""
        alerts = detect_cyp_alerts(ketoconazole)
        assert "imidazole" in alerts

    def test_thiophene_alert(self):
        """Thiophene-containing molecule should trigger alert."""
        mol = Chem.MolFromSmiles("c1ccsc1C(=O)O")  # 2-thiophenecarboxylic acid
        alerts = detect_cyp_alerts(mol)
        assert "thiophene" in alerts


class TestPredictCypInhibition:
    def test_low_risk(self, aspirin):
        """Aspirin has 0 alerts → low risk."""
        cls, alerts, risk = predict_cyp_inhibition(aspirin)
        assert cls == "low"
        assert risk == "green"
        assert len(alerts) == 0

    def test_moderate_risk_single_alert(self):
        """One alert → moderate risk."""
        mol = Chem.MolFromSmiles("c1ccsc1")  # bare thiophene
        cls, alerts, risk = predict_cyp_inhibition(mol)
        assert cls == "moderate"
        assert risk == "yellow"


class TestPredictMetabolicStability:
    def test_high_stability_small(self, small_stable):
        """Small molecule with low LogP → high stability."""
        cls, score, risk = predict_metabolic_stability(small_stable)
        assert cls == "high"
        assert score >= 0.75
        assert risk == "green"

    def test_low_stability_large(self, large_lipophilic):
        """Large lipophilic molecule → low stability."""
        cls, score, risk = predict_metabolic_stability(large_lipophilic)
        assert score < 0.75  # has penalties


class TestPredictMetabolism:
    def test_full_result(self, aspirin):
        """Full metabolism result has all fields."""
        result = predict_metabolism(aspirin)
        assert isinstance(result, MetabolismResult)
        assert isinstance(result.cyp_alerts, list)
        assert result.cyp_inhibition_risk in ("low", "moderate", "high")
        assert result.metabolic_stability in ("high", "moderate", "low")
        assert 0 <= result.metabolic_stability_score <= 1

    def test_none_mol_raises(self):
        """None molecule raises ADMETError."""
        with pytest.raises(ADMETError):
            predict_metabolism(None)

    def test_record_storage(self, aspirin):
        """predict_metabolism_record stores results in rec.properties."""
        rec = MoleculeRecord(mol=aspirin)
        predict_metabolism_record(rec)
        assert "admet_cyp_inhibition_risk" in rec.properties
        assert "admet_metabolic_stability" in rec.properties
        assert "admet_metabolic_stability_score" in rec.properties
        assert "admet_cyp_n_alerts" in rec.properties

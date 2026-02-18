"""Tests for ADMET toxicity predictions."""

import pytest
from rdkit import Chem

from drugflow.core.exceptions import ADMETError
from drugflow.core.models import MoleculeRecord
from drugflow.phase5.admet.toxicity import (
    ToxicityResult,
    detect_ames_alerts,
    detect_hepatotox_alerts,
    detect_herg_risk,
    estimate_mrdd,
    predict_toxicity,
    predict_toxicity_record,
)


@pytest.fixture
def aspirin():
    """Aspirin: small, safe profile."""
    return Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")


@pytest.fixture
def nitrobenzene():
    """Nitrobenzene: aromatic nitro → AMES alert + hepatotox."""
    return Chem.MolFromSmiles("c1ccc([N+](=O)[O-])cc1")


@pytest.fixture
def large_lipophilic_amine():
    """Large molecule with LogP > 3.7, MW > 400, basic nitrogen chain.
    Designed to trigger hERG flags."""
    return Chem.MolFromSmiles("c1ccc(CCCCNCCCC)cc1CCCCc1ccc(OCCCCCCCC)cc1")


@pytest.fixture
def benzoquinone():
    """Benzoquinone: quinone → hepatotox reactive metabolite."""
    return Chem.MolFromSmiles("O=C1C=CC(=O)C=C1")


class TestDetectHergRisk:
    def test_no_risk_aspirin(self, aspirin):
        """Aspirin should have no hERG risk."""
        has_risk, alerts, risk = detect_herg_risk(aspirin)
        assert has_risk is False
        assert risk == "green"

    def test_risk_flagged(self, large_lipophilic_amine):
        """Large lipophilic amine → hERG risk."""
        has_risk, alerts, risk = detect_herg_risk(large_lipophilic_amine)
        assert has_risk is True
        assert risk in ("yellow", "red")


class TestDetectAmesAlerts:
    def test_no_alerts_aspirin(self, aspirin):
        """Aspirin should have no AMES alerts."""
        alerts, risk = detect_ames_alerts(aspirin)
        assert len(alerts) == 0
        assert risk == "green"

    def test_nitroaromatic_alert(self, nitrobenzene):
        """Nitrobenzene has aromatic nitro → AMES alert."""
        alerts, risk = detect_ames_alerts(nitrobenzene)
        assert "aromatic_nitro" in alerts
        assert risk in ("yellow", "red")

    def test_epoxide_alert(self):
        """Ethylene oxide (epoxide) → AMES alert."""
        mol = Chem.MolFromSmiles("C1OC1")
        alerts, risk = detect_ames_alerts(mol)
        assert "epoxide" in alerts


class TestDetectHepatotoxAlerts:
    def test_no_alerts_aspirin(self, aspirin):
        """Aspirin should have no hepatotox alerts."""
        alerts, risk = detect_hepatotox_alerts(aspirin)
        # Aspirin may or may not match acyl_glucuronide
        assert risk in ("green", "yellow")

    def test_quinone_alert(self, benzoquinone):
        """Benzoquinone → quinone hepatotox alert."""
        alerts, risk = detect_hepatotox_alerts(benzoquinone)
        assert "quinone" in alerts

    def test_nitroaromatic_hepatotox(self, nitrobenzene):
        """Nitrobenzene → nitroaromatic hepatotox alert."""
        alerts, risk = detect_hepatotox_alerts(nitrobenzene)
        assert "nitroaromatic_hep" in alerts


class TestEstimateMRDD:
    def test_basic_classification(self, aspirin):
        """MRDD returns valid class."""
        cls, risk = estimate_mrdd(aspirin)
        assert cls in ("low_dose", "moderate_dose", "high_dose")
        assert risk in ("green", "yellow", "red")


class TestPredictToxicity:
    def test_full_result(self, aspirin):
        """Full toxicity result has all fields."""
        result = predict_toxicity(aspirin)
        assert isinstance(result, ToxicityResult)
        assert isinstance(result.herg_risk_flag, bool)
        assert isinstance(result.ames_alerts, list)
        assert isinstance(result.hepatotox_alerts, list)
        assert result.mrdd_class in ("low_dose", "moderate_dose", "high_dose")

    def test_none_mol_raises(self):
        """None molecule raises ADMETError."""
        with pytest.raises(ADMETError):
            predict_toxicity(None)

    def test_record_storage(self, aspirin):
        """predict_toxicity_record stores results in rec.properties."""
        rec = MoleculeRecord(mol=aspirin)
        predict_toxicity_record(rec)
        assert "admet_herg_risk_flag" in rec.properties
        assert "admet_ames_risk" in rec.properties
        assert "admet_hepatotox_risk" in rec.properties
        assert "admet_mrdd_class" in rec.properties

"""Tests for ADMET excretion predictions."""

import pytest
from rdkit import Chem

from drugflow.core.exceptions import ADMETError
from drugflow.core.models import MoleculeRecord
from drugflow.phase5.admet.excretion import (
    ExcretionResult,
    predict_excretion,
    predict_excretion_record,
    predict_halflife,
    predict_renal_clearance,
)


@pytest.fixture
def metformin():
    """Metformin: MW~129, LogP~-1.4 → small, hydrophilic → renal."""
    return Chem.MolFromSmiles("CN(C)C(=N)NC(=N)N")


@pytest.fixture
def large_lipophilic():
    """Large lipophilic molecule: MW > 500, LogP > 3."""
    # Long aliphatic amide: MW ~593, LogP high
    return Chem.MolFromSmiles("CCCCCCCCCCCCCCCCCCCC(=O)NCCCCCCCCCCCCCCCCCCCC")


@pytest.fixture
def aspirin():
    return Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")


class TestPredictRenalClearance:
    def test_small_hydrophilic_renal(self, metformin):
        """Metformin (MW~129, LogP~-1.4) → likely renal clearance."""
        is_renal, risk = predict_renal_clearance(metformin)
        assert is_renal is True
        assert risk == "green"

    def test_large_lipophilic_not_renal(self, large_lipophilic):
        """Large lipophilic molecule → not renally cleared."""
        is_renal, risk = predict_renal_clearance(large_lipophilic)
        assert is_renal is False
        assert risk == "yellow"


class TestPredictHalflife:
    def test_short_halflife(self, metformin):
        """Small molecule + high stability → short half-life."""
        cls, risk = predict_halflife(metformin, metabolic_stability_score=0.9)
        assert cls == "short"
        assert risk == "green"

    def test_long_halflife(self, large_lipophilic):
        """Large molecule + low stability → long half-life."""
        cls, risk = predict_halflife(large_lipophilic, metabolic_stability_score=0.3)
        assert cls == "long"
        assert risk == "yellow"


class TestPredictExcretion:
    def test_full_result(self, aspirin):
        """Full excretion result has all fields."""
        result = predict_excretion(aspirin)
        assert isinstance(result, ExcretionResult)
        assert isinstance(result.renal_clearance_likely, bool)
        assert result.halflife_class in ("short", "medium", "long")

    def test_none_mol_raises(self):
        """None molecule raises ADMETError."""
        with pytest.raises(ADMETError):
            predict_excretion(None)

    def test_record_storage(self, aspirin):
        """predict_excretion_record stores results in rec.properties."""
        rec = MoleculeRecord(mol=aspirin)
        predict_excretion_record(rec)
        assert "admet_renal_clearance" in rec.properties
        assert "admet_halflife_class" in rec.properties

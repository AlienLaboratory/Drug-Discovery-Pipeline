"""Tests for ADMET pipeline (batch prediction)."""

import pytest
from rdkit import Chem

from drugflow.core.exceptions import ADMETError
from drugflow.core.models import MoleculeDataset, MoleculeRecord
from drugflow.phase5.admet.pipeline import predict_admet, predict_admet_dataset


@pytest.fixture
def sample_dataset():
    """Small dataset with 3 molecules."""
    records = []
    smiles_list = [
        "CC(=O)Oc1ccccc1C(=O)O",   # aspirin
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",  # caffeine
        "c1ccccc1",  # benzene
    ]
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        rec = MoleculeRecord(mol=mol, record_id=f"mol_{i}")
        records.append(rec)
    return MoleculeDataset(records=records)


class TestPredictAdmetSingle:
    def test_returns_dict(self):
        """Single molecule returns dict with admet_ keys."""
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        result = predict_admet(mol)
        assert isinstance(result, dict)
        assert all(k.startswith("admet_") for k in result.keys())

    def test_has_all_expected_keys(self):
        """Result contains all major ADMET keys."""
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        result = predict_admet(mol)
        expected = [
            "admet_score", "admet_class",
            "admet_caco2_class", "admet_hia_class",
            "admet_bbb_penetrant", "admet_ppb_class",
            "admet_cyp_inhibition_risk", "admet_metabolic_stability",
            "admet_renal_clearance", "admet_halflife_class",
            "admet_herg_risk_flag", "admet_ames_risk",
        ]
        for key in expected:
            assert key in result, f"Missing key: {key}"

    def test_none_raises(self):
        """None molecule raises ADMETError."""
        with pytest.raises(ADMETError):
            predict_admet(None)


class TestPredictAdmetDataset:
    def test_batch_prediction(self, sample_dataset):
        """Batch prediction populates all records."""
        result = predict_admet_dataset(sample_dataset)
        assert result is sample_dataset  # returns same dataset

        for rec in result.valid_records:
            assert "admet_score" in rec.properties
            assert "admet_class" in rec.properties

    def test_provenance_added(self, sample_dataset):
        """Provenance 'admet:predicted' is added."""
        predict_admet_dataset(sample_dataset)
        for rec in sample_dataset.valid_records:
            assert "admet:predicted" in rec.provenance

    def test_handles_invalid_gracefully(self):
        """Dataset with None mol doesn't crash."""
        good = MoleculeRecord(mol=Chem.MolFromSmiles("CCO"), record_id="good")
        bad = MoleculeRecord(mol=None, record_id="bad")
        dataset = MoleculeDataset(records=[good, bad])

        result = predict_admet_dataset(dataset)
        # Good record should have predictions
        assert "admet_score" in good.properties

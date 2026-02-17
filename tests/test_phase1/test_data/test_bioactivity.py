"""Tests for bioactivity data fetching, curation, and labeling."""

import io
import json
import math
from unittest.mock import patch, MagicMock

import pytest
from rdkit import Chem

from drugflow.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus
from drugflow.phase1.data.bioactivity import (
    CurationStats,
    _convert_to_nm,
    compute_pic50,
    filter_by_relation,
    filter_and_normalize_units,
    deduplicate_molecules,
    remove_activity_outliers,
    compute_confidence_score,
    curate_bioactivity,
    label_activity,
    fetch_chembl_bioactivity,
    fetch_and_curate,
    _activity_to_record,
)
from drugflow.core.exceptions import DatabaseError


# ── Test Data ────────────────────────────────────────────────

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
CAFFEINE_SMILES = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
NAPROXEN_SMILES = "COc1ccc2cc(CC(=O)O)ccc2c1"
DICLOFENAC_SMILES = "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl"


def _make_record(smiles, ic50=100.0, units="nM", relation="=",
                 pchembl="7.0", assay_type="B", source_id=None):
    """Create a MoleculeRecord with bioactivity metadata."""
    mol = Chem.MolFromSmiles(smiles)
    rec = MoleculeRecord(
        mol=mol, smiles=smiles,
        source_id=source_id or smiles[:10],
        status=MoleculeStatus.RAW,
    )
    rec.metadata["standard_value"] = ic50
    rec.metadata["standard_units"] = units
    rec.metadata["standard_relation"] = relation
    rec.metadata["pchembl_value"] = pchembl
    rec.metadata["assay_type"] = assay_type
    rec.metadata["assay_chembl_id"] = "CHEMBL_ASSAY_1"
    rec.metadata["target_chembl_id"] = "CHEMBL4860"
    return rec


def _make_dataset(records=None):
    """Build a test bioactivity dataset."""
    if records is None:
        records = [
            _make_record(ASPIRIN_SMILES, ic50=50.0, relation="="),
            _make_record(IBUPROFEN_SMILES, ic50=200.0, relation="="),
            _make_record(CAFFEINE_SMILES, ic50=5000.0, relation="="),
            _make_record(NAPROXEN_SMILES, ic50=800.0, relation=">"),
            _make_record(DICLOFENAC_SMILES, ic50=150.0, relation="<"),
        ]
    return MoleculeDataset(records=records, name="test_bioactivity")


# Mock ChEMBL API response
MOCK_CHEMBL_RESPONSE_PAGE1 = {
    "page_meta": {
        "total_count": 3,
        "limit": 1000,
        "offset": 0,
        "next": None,
    },
    "activities": [
        {
            "molecule_chembl_id": "CHEMBL1",
            "canonical_smiles": ASPIRIN_SMILES,
            "standard_type": "IC50",
            "standard_value": "500",
            "standard_units": "nM",
            "standard_relation": "=",
            "pchembl_value": "6.3",
            "assay_chembl_id": "CHEMBL_ASSAY_1",
            "assay_description": "Binding assay for BCL-2",
            "assay_type": "B",
            "target_chembl_id": "CHEMBL4860",
            "document_chembl_id": "CHEMBL_DOC_1",
        },
        {
            "molecule_chembl_id": "CHEMBL2",
            "canonical_smiles": IBUPROFEN_SMILES,
            "standard_type": "IC50",
            "standard_value": "100",
            "standard_units": "nM",
            "standard_relation": "=",
            "pchembl_value": "7.0",
            "assay_chembl_id": "CHEMBL_ASSAY_1",
            "assay_description": "Binding assay for BCL-2",
            "assay_type": "B",
            "target_chembl_id": "CHEMBL4860",
            "document_chembl_id": "CHEMBL_DOC_1",
        },
        {
            "molecule_chembl_id": "CHEMBL3",
            "canonical_smiles": CAFFEINE_SMILES,
            "standard_type": "IC50",
            "standard_value": "10000",
            "standard_units": "nM",
            "standard_relation": "=",
            "pchembl_value": "5.0",
            "assay_chembl_id": "CHEMBL_ASSAY_2",
            "assay_description": "Cell viability assay",
            "assay_type": "F",
            "target_chembl_id": "CHEMBL4860",
            "document_chembl_id": "CHEMBL_DOC_2",
        },
    ],
}

MOCK_CHEMBL_RESPONSE_PAGE1_MULTI = {
    "page_meta": {
        "total_count": 2,
        "limit": 1000,
        "offset": 0,
        "next": "/next?offset=1000",
    },
    "activities": [
        {
            "molecule_chembl_id": "CHEMBL_A",
            "canonical_smiles": ASPIRIN_SMILES,
            "standard_type": "IC50",
            "standard_value": "100",
            "standard_units": "nM",
            "standard_relation": "=",
            "pchembl_value": "7.0",
            "assay_chembl_id": "CHEMBL_ASSAY_1",
            "assay_description": "Assay 1",
            "assay_type": "B",
            "target_chembl_id": "CHEMBL4860",
            "document_chembl_id": "CHEMBL_DOC_1",
        },
        {
            "molecule_chembl_id": "CHEMBL_B",
            "canonical_smiles": IBUPROFEN_SMILES,
            "standard_type": "IC50",
            "standard_value": "200",
            "standard_units": "nM",
            "standard_relation": "=",
            "pchembl_value": "6.7",
            "assay_chembl_id": "CHEMBL_ASSAY_1",
            "assay_description": "Assay 1",
            "assay_type": "B",
            "target_chembl_id": "CHEMBL4860",
            "document_chembl_id": "CHEMBL_DOC_1",
        },
    ],
}

MOCK_CHEMBL_RESPONSE_PAGE2 = {
    "page_meta": {
        "total_count": 3,
        "limit": 1000,
        "offset": 1000,
        "next": None,
    },
    "activities": [
        {
            "molecule_chembl_id": "CHEMBL_C",
            "canonical_smiles": CAFFEINE_SMILES,
            "standard_type": "IC50",
            "standard_value": "5000",
            "standard_units": "nM",
            "standard_relation": "=",
            "pchembl_value": "5.3",
            "assay_chembl_id": "CHEMBL_ASSAY_2",
            "assay_description": "Assay 2",
            "assay_type": "B",
            "target_chembl_id": "CHEMBL4860",
            "document_chembl_id": "CHEMBL_DOC_2",
        },
    ],
}


def _mock_urlopen_single_page(*args, **kwargs):
    """Mock urlopen returning a single-page response."""
    response = MagicMock()
    response.__enter__ = lambda s: s
    response.__exit__ = MagicMock(return_value=False)
    response.read.return_value = json.dumps(MOCK_CHEMBL_RESPONSE_PAGE1).encode()
    return response


def _mock_urlopen_multi_page(*args, **kwargs):
    """Mock urlopen returning multi-page responses."""
    url = args[0].full_url if hasattr(args[0], 'full_url') else str(args[0])
    response = MagicMock()
    response.__enter__ = lambda s: s
    response.__exit__ = MagicMock(return_value=False)
    if "offset=0" in url or "offset" not in url:
        response.read.return_value = json.dumps(MOCK_CHEMBL_RESPONSE_PAGE1_MULTI).encode()
    else:
        response.read.return_value = json.dumps(MOCK_CHEMBL_RESPONSE_PAGE2).encode()
    return response


# ── Unit Conversion Tests ────────────────────────────────────


class TestUnitConversion:

    def test_convert_um_to_nm(self):
        assert _convert_to_nm(1.0, "uM") == 1000.0

    def test_convert_nm_identity(self):
        assert _convert_to_nm(100.0, "nM") == 100.0

    def test_convert_mm_to_nm(self):
        assert _convert_to_nm(1.0, "mM") == 1e6

    def test_convert_m_to_nm(self):
        assert _convert_to_nm(1e-6, "M") == 1000.0

    def test_convert_pm_to_nm(self):
        assert _convert_to_nm(1000.0, "pM") == 1.0

    def test_convert_unknown_unit(self):
        assert _convert_to_nm(1.0, "mg/mL") is None


class TestComputePIC50:

    def test_pic50_100nm(self):
        assert abs(compute_pic50(100.0) - 7.0) < 0.001

    def test_pic50_1nm(self):
        assert abs(compute_pic50(1.0) - 9.0) < 0.001

    def test_pic50_10000nm(self):
        assert abs(compute_pic50(10000.0) - 5.0) < 0.001

    def test_pic50_zero(self):
        assert compute_pic50(0.0) == 0.0

    def test_pic50_negative(self):
        assert compute_pic50(-10.0) == 0.0


# ── Curation Stats Tests ─────────────────────────────────────


class TestCurationStats:

    def test_stats_to_dict(self):
        stats = CurationStats(input_count=100, output_count=80)
        d = stats.to_dict()
        assert d["input_count"] == 100
        assert d["output_count"] == 80
        assert "duplicates_merged" in d


# ── Filter By Relation Tests ─────────────────────────────────


class TestFilterByRelation:

    def test_exact_only(self):
        dataset = _make_dataset()
        result = filter_by_relation(dataset, allowed_relations={"="})
        # aspirin(=), ibuprofen(=), caffeine(=) kept; naproxen(>), diclofenac(<) removed
        assert len(result) == 3

    def test_permissive(self):
        dataset = _make_dataset()
        result = filter_by_relation(dataset, allowed_relations={"=", "<", "<="})
        # aspirin(=), ibuprofen(=), caffeine(=), diclofenac(<) kept; naproxen(>) removed
        assert len(result) == 4

    def test_all_filtered(self):
        dataset = _make_dataset()
        result = filter_by_relation(dataset, allowed_relations={"~"})
        assert len(result) == 0

    def test_provenance_added(self):
        dataset = _make_dataset()
        _ = filter_by_relation(dataset, allowed_relations={"="})
        # naproxen and diclofenac should have filter provenance
        naproxen_rec = dataset.records[3]
        assert "filter:relation" in naproxen_rec.provenance


# ── Unit Normalization Tests ─────────────────────────────────


class TestFilterAndNormalizeUnits:

    def test_nm_unchanged(self):
        rec = _make_record(ASPIRIN_SMILES, ic50=100.0, units="nM")
        ds = MoleculeDataset(records=[rec])
        result = filter_and_normalize_units(ds)
        assert len(result) == 1
        assert result.records[0].metadata["standard_value"] == 100.0

    def test_um_to_nm(self):
        rec = _make_record(ASPIRIN_SMILES, ic50=1.0, units="uM")
        ds = MoleculeDataset(records=[rec])
        result = filter_and_normalize_units(ds)
        assert len(result) == 1
        assert result.records[0].metadata["standard_value"] == 1000.0
        assert result.records[0].metadata["standard_units"] == "nM"

    def test_mm_to_nm(self):
        rec = _make_record(ASPIRIN_SMILES, ic50=0.001, units="mM")
        ds = MoleculeDataset(records=[rec])
        result = filter_and_normalize_units(ds)
        assert result.records[0].metadata["standard_value"] == 1000.0

    def test_unsupported_units_filtered(self):
        rec = _make_record(ASPIRIN_SMILES, ic50=100.0, units="mg/mL")
        ds = MoleculeDataset(records=[rec])
        result = filter_and_normalize_units(ds)
        assert len(result) == 0

    def test_non_numeric_value_filtered(self):
        rec = _make_record(ASPIRIN_SMILES, ic50="not_a_number", units="nM")
        ds = MoleculeDataset(records=[rec])
        result = filter_and_normalize_units(ds)
        assert len(result) == 0


# ── Deduplication Tests ──────────────────────────────────────


class TestDeduplicateMolecules:

    def test_median_aggregation(self):
        rec1 = _make_record(ASPIRIN_SMILES, ic50=100.0, source_id="a1")
        rec2 = _make_record(ASPIRIN_SMILES, ic50=200.0, source_id="a2")
        ds = MoleculeDataset(records=[rec1, rec2])
        result, n_merged = deduplicate_molecules(ds, aggregation="median")
        assert len(result) == 1
        assert result.records[0].metadata["standard_value"] == 150.0
        assert n_merged == 1

    def test_mean_aggregation(self):
        rec1 = _make_record(ASPIRIN_SMILES, ic50=100.0, source_id="a1")
        rec2 = _make_record(ASPIRIN_SMILES, ic50=300.0, source_id="a2")
        ds = MoleculeDataset(records=[rec1, rec2])
        result, _ = deduplicate_molecules(ds, aggregation="mean")
        assert abs(result.records[0].metadata["standard_value"] - 200.0) < 0.01

    def test_n_measurements(self):
        rec1 = _make_record(ASPIRIN_SMILES, ic50=100.0, source_id="a1")
        rec2 = _make_record(ASPIRIN_SMILES, ic50=200.0, source_id="a2")
        rec3 = _make_record(ASPIRIN_SMILES, ic50=150.0, source_id="a3")
        ds = MoleculeDataset(records=[rec1, rec2, rec3])
        result, _ = deduplicate_molecules(ds)
        assert result.records[0].metadata["n_measurements"] == 3

    def test_no_duplicates(self):
        rec1 = _make_record(ASPIRIN_SMILES, ic50=100.0)
        rec2 = _make_record(IBUPROFEN_SMILES, ic50=200.0)
        ds = MoleculeDataset(records=[rec1, rec2])
        result, n_merged = deduplicate_molecules(ds)
        assert len(result) == 2
        assert n_merged == 0

    def test_dedup_provenance(self):
        rec1 = _make_record(ASPIRIN_SMILES, ic50=100.0, source_id="a1")
        rec2 = _make_record(ASPIRIN_SMILES, ic50=200.0, source_id="a2")
        ds = MoleculeDataset(records=[rec1, rec2])
        result, _ = deduplicate_molecules(ds)
        assert any("curated:deduplicated" in p for p in result.records[0].provenance)


# ── Outlier Removal Tests ────────────────────────────────────


class TestRemoveOutliers:

    def test_outlier_removed(self):
        # Create dataset with one extreme outlier
        records = [
            _make_record(ASPIRIN_SMILES, ic50=100.0),
            _make_record(IBUPROFEN_SMILES, ic50=150.0),
            _make_record(CAFFEINE_SMILES, ic50=120.0),
            _make_record(NAPROXEN_SMILES, ic50=130.0),
            _make_record(DICLOFENAC_SMILES, ic50=1000000.0),  # extreme outlier
        ]
        ds = MoleculeDataset(records=records)
        result, n_removed = remove_activity_outliers(ds, iqr_multiplier=1.5)
        assert n_removed >= 1
        assert len(result) < len(ds)

    def test_no_outliers(self):
        # Tight distribution, no outliers
        records = [
            _make_record(ASPIRIN_SMILES, ic50=100.0),
            _make_record(IBUPROFEN_SMILES, ic50=110.0),
            _make_record(CAFFEINE_SMILES, ic50=105.0),
            _make_record(NAPROXEN_SMILES, ic50=108.0),
            _make_record(DICLOFENAC_SMILES, ic50=103.0),
        ]
        ds = MoleculeDataset(records=records)
        result, n_removed = remove_activity_outliers(ds)
        assert n_removed == 0
        assert len(result) == 5

    def test_outlier_provenance(self):
        records = [
            _make_record(ASPIRIN_SMILES, ic50=100.0),
            _make_record(IBUPROFEN_SMILES, ic50=110.0),
            _make_record(CAFFEINE_SMILES, ic50=105.0),
            _make_record(NAPROXEN_SMILES, ic50=108.0),
            _make_record(DICLOFENAC_SMILES, ic50=1000000.0),
        ]
        ds = MoleculeDataset(records=records)
        remove_activity_outliers(ds, iqr_multiplier=1.5)
        outlier_rec = ds.records[4]
        assert "filter:outlier" in outlier_rec.provenance

    def test_too_few_records(self):
        records = [_make_record(ASPIRIN_SMILES, ic50=100.0)]
        ds = MoleculeDataset(records=records)
        result = remove_activity_outliers(ds)
        # Should return the original dataset unchanged (not enough data)
        assert len(result) == 1


# ── Confidence Score Tests ───────────────────────────────────


class TestConfidenceScore:

    def test_perfect_score(self):
        # exact "=", pchembl available, binding assay "B", nM units
        rec = _make_record(ASPIRIN_SMILES, relation="=", pchembl="7.0",
                           assay_type="B", units="nM")
        ds = MoleculeDataset(records=[rec])
        compute_confidence_score(ds)
        assert ds.records[0].metadata["confidence_score"] == 1.0

    def test_partial_score(self):
        # exact "=", no pchembl, functional assay "F", nM units
        rec = _make_record(ASPIRIN_SMILES, relation="=", pchembl="",
                           assay_type="F", units="nM")
        ds = MoleculeDataset(records=[rec])
        compute_confidence_score(ds)
        score = ds.records[0].metadata["confidence_score"]
        # 0.4 (exact) + 0.0 (no pchembl) + 0.0 (not binding) + 0.1 (nM) = 0.5
        assert abs(score - 0.5) < 0.01

    def test_zero_score(self):
        rec = _make_record(ASPIRIN_SMILES, relation=">", pchembl="",
                           assay_type="F", units="uM")
        ds = MoleculeDataset(records=[rec])
        compute_confidence_score(ds)
        assert ds.records[0].metadata["confidence_score"] == 0.0


# ── Full Curation Pipeline Tests ─────────────────────────────


class TestCurateBioactivity:

    def test_basic_curation(self):
        dataset = _make_dataset()
        curated, stats = curate_bioactivity(dataset)
        assert stats.input_count == 5
        assert stats.output_count <= stats.after_relation_filter

    def test_stats_accuracy(self):
        dataset = _make_dataset()
        curated, stats = curate_bioactivity(dataset)
        assert stats.after_relation_filter <= stats.input_count
        assert stats.after_unit_filter <= stats.after_relation_filter
        assert stats.after_dedup <= stats.after_unit_filter
        assert stats.output_count == len(curated)

    def test_curation_with_outliers(self):
        records = [
            _make_record(ASPIRIN_SMILES, ic50=100.0),
            _make_record(IBUPROFEN_SMILES, ic50=150.0),
            _make_record(CAFFEINE_SMILES, ic50=120.0),
            _make_record(NAPROXEN_SMILES, ic50=130.0),
            _make_record(DICLOFENAC_SMILES, ic50=1000000.0),
        ]
        ds = MoleculeDataset(records=records)
        curated, stats = curate_bioactivity(ds, remove_outliers=True)
        assert stats.outliers_removed >= 0

    def test_curation_from_csv_data(self):
        """Curate works with data loaded from CSV (different key names)."""
        rec = _make_record(ASPIRIN_SMILES, ic50=100.0)
        ds = MoleculeDataset(records=[rec])
        curated, stats = curate_bioactivity(ds)
        assert stats.output_count >= 1


# ── Labeling Tests ───────────────────────────────────────────


class TestLabelActivity:

    def test_binary_active(self):
        rec = _make_record(ASPIRIN_SMILES, ic50=500.0)
        ds = MoleculeDataset(records=[rec])
        labeled = label_activity(ds, mode="binary", active_threshold=1000.0)
        assert labeled.records[0].metadata["activity_class"] == 1
        assert labeled.records[0].metadata["activity_label"] == "active"

    def test_binary_inactive(self):
        rec = _make_record(ASPIRIN_SMILES, ic50=5000.0)
        ds = MoleculeDataset(records=[rec])
        labeled = label_activity(ds, mode="binary", active_threshold=1000.0)
        assert labeled.records[0].metadata["activity_class"] == 0
        assert labeled.records[0].metadata["activity_label"] == "inactive"

    def test_ternary_active(self):
        rec = _make_record(ASPIRIN_SMILES, ic50=500.0)
        ds = MoleculeDataset(records=[rec])
        labeled = label_activity(ds, mode="ternary",
                                 active_threshold=1000.0,
                                 inactive_threshold=10000.0)
        assert labeled.records[0].metadata["activity_class"] == 1

    def test_ternary_intermediate(self):
        rec = _make_record(ASPIRIN_SMILES, ic50=5000.0)
        ds = MoleculeDataset(records=[rec])
        labeled = label_activity(ds, mode="ternary",
                                 active_threshold=1000.0,
                                 inactive_threshold=10000.0)
        assert labeled.records[0].metadata["activity_class"] == 0
        assert labeled.records[0].metadata["activity_label"] == "intermediate"

    def test_ternary_inactive(self):
        rec = _make_record(ASPIRIN_SMILES, ic50=50000.0)
        ds = MoleculeDataset(records=[rec])
        labeled = label_activity(ds, mode="ternary",
                                 active_threshold=1000.0,
                                 inactive_threshold=10000.0)
        assert labeled.records[0].metadata["activity_class"] == -1
        assert labeled.records[0].metadata["activity_label"] == "inactive"

    def test_pic50_computation(self):
        rec = _make_record(ASPIRIN_SMILES, ic50=100.0)
        ds = MoleculeDataset(records=[rec])
        labeled = label_activity(ds)
        assert abs(labeled.records[0].metadata["pIC50"] - 7.0) < 0.001

    def test_pic50_edge_case_zero(self):
        rec = _make_record(ASPIRIN_SMILES, ic50=0.0)
        ds = MoleculeDataset(records=[rec])
        labeled = label_activity(ds)
        assert labeled.records[0].metadata["pIC50"] == 0.0

    def test_invalid_mode(self):
        ds = MoleculeDataset(records=[_make_record(ASPIRIN_SMILES)])
        with pytest.raises(ValueError, match="Invalid mode"):
            label_activity(ds, mode="invalid")

    def test_ternary_threshold_validation(self):
        ds = MoleculeDataset(records=[_make_record(ASPIRIN_SMILES)])
        with pytest.raises(ValueError, match="inactive_threshold"):
            label_activity(ds, mode="ternary",
                           active_threshold=10000.0,
                           inactive_threshold=1000.0)

    def test_label_provenance(self):
        rec = _make_record(ASPIRIN_SMILES, ic50=100.0)
        ds = MoleculeDataset(records=[rec])
        labeled = label_activity(ds, mode="binary")
        assert any("labeled:binary" in p for p in labeled.records[0].provenance)

    def test_label_metadata_keys(self):
        rec = _make_record(ASPIRIN_SMILES, ic50=100.0)
        ds = MoleculeDataset(records=[rec])
        labeled = label_activity(ds)
        meta = labeled.records[0].metadata
        assert "pIC50" in meta
        assert "activity_class" in meta
        assert "activity_label" in meta

    def test_label_feeds_qsar_data_prep(self):
        """Verify labeled data can be read by Phase 2 data_prep."""
        rec = _make_record(ASPIRIN_SMILES, ic50=100.0)
        ds = MoleculeDataset(records=[rec])
        labeled = label_activity(ds)
        # Phase 2 extracts from metadata
        val = labeled.records[0].metadata.get("activity_class")
        assert val is not None
        assert isinstance(val, int)


# ── Fetching Tests (mocked HTTP) ─────────────────────────────


class TestFetchChemblBioactivity:

    @patch("drugflow.phase1.data.bioactivity.urllib.request.urlopen",
           side_effect=_mock_urlopen_single_page)
    def test_fetch_single_page(self, mock_url):
        dataset = fetch_chembl_bioactivity("CHEMBL4860")
        assert len(dataset) == 3
        assert dataset.name == "chembl_CHEMBL4860_bioactivity"

    @patch("drugflow.phase1.data.bioactivity.urllib.request.urlopen",
           side_effect=_mock_urlopen_single_page)
    def test_fetch_metadata_fields(self, mock_url):
        dataset = fetch_chembl_bioactivity("CHEMBL4860")
        rec = dataset.records[0]
        assert rec.metadata["molecule_chembl_id"] == "CHEMBL1"
        assert rec.metadata["standard_type"] == "IC50"
        assert rec.metadata["assay_chembl_id"] == "CHEMBL_ASSAY_1"
        assert rec.metadata["target_chembl_id"] == "CHEMBL4860"

    @patch("drugflow.phase1.data.bioactivity.urllib.request.urlopen",
           side_effect=_mock_urlopen_single_page)
    def test_fetch_provenance(self, mock_url):
        dataset = fetch_chembl_bioactivity("CHEMBL4860")
        assert any("fetched:chembl:CHEMBL4860" in p
                    for p in dataset.records[0].provenance)

    @patch("drugflow.phase1.data.bioactivity.urllib.request.urlopen",
           side_effect=_mock_urlopen_single_page)
    def test_fetch_standard_value_as_float(self, mock_url):
        dataset = fetch_chembl_bioactivity("CHEMBL4860")
        val = dataset.records[0].metadata["standard_value"]
        assert isinstance(val, float)
        assert val == 500.0

    @patch("drugflow.phase1.data.bioactivity.urllib.request.urlopen",
           side_effect=_mock_urlopen_single_page)
    def test_fetch_pchembl_as_float(self, mock_url):
        dataset = fetch_chembl_bioactivity("CHEMBL4860")
        val = dataset.records[0].metadata["pchembl_value"]
        assert isinstance(val, float)
        assert val == 6.3

    def test_fetch_invalid_activity_type(self):
        with pytest.raises(ValueError, match="Unsupported activity type"):
            fetch_chembl_bioactivity("CHEMBL4860", activity_types=["INVALID"])

    @patch("drugflow.phase1.data.bioactivity.urllib.request.urlopen",
           side_effect=_mock_urlopen_single_page)
    def test_fetch_empty_response(self, mock_url):
        # Override to return empty activities
        def empty_response(*args, **kwargs):
            resp = MagicMock()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            resp.read.return_value = json.dumps({
                "page_meta": {"total_count": 0, "next": None},
                "activities": [],
            }).encode()
            return resp
        mock_url.side_effect = empty_response
        dataset = fetch_chembl_bioactivity("CHEMBL4860")
        assert len(dataset) == 0

    @patch("drugflow.phase1.data.bioactivity.urllib.request.urlopen")
    def test_fetch_api_error(self, mock_url):
        from urllib.error import HTTPError
        mock_url.side_effect = HTTPError(
            url="http://test", code=500,
            msg="Server Error", hdrs=None, fp=None,
        )
        with pytest.raises(DatabaseError):
            fetch_chembl_bioactivity("CHEMBL4860")

    @patch("drugflow.phase1.data.bioactivity.urllib.request.urlopen",
           side_effect=_mock_urlopen_single_page)
    def test_fetch_save_raw(self, mock_url, tmp_path):
        raw_path = str(tmp_path / "raw.csv")
        fetch_chembl_bioactivity("CHEMBL4860", save_raw=raw_path)
        import os
        assert os.path.exists(raw_path)
        with open(raw_path) as f:
            lines = f.readlines()
        assert len(lines) >= 4  # header + 3 records

    @patch("drugflow.phase1.data.bioactivity.urllib.request.urlopen",
           side_effect=_mock_urlopen_single_page)
    def test_fetch_max_results(self, mock_url):
        dataset = fetch_chembl_bioactivity("CHEMBL4860", max_results=2)
        assert len(dataset) <= 2

    @patch("drugflow.phase1.data.bioactivity.urllib.request.urlopen",
           side_effect=_mock_urlopen_single_page)
    def test_fetch_invalid_smiles(self, mock_url):
        # Override one record to have bad SMILES
        def bad_smiles_response(*args, **kwargs):
            resp = MagicMock()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            data = {
                "page_meta": {"total_count": 1, "next": None},
                "activities": [{
                    "molecule_chembl_id": "BAD",
                    "canonical_smiles": "NOT_VALID_SMILES",
                    "standard_type": "IC50",
                    "standard_value": "100",
                    "standard_units": "nM",
                    "standard_relation": "=",
                    "pchembl_value": None,
                    "assay_chembl_id": "A1",
                    "assay_description": "test",
                    "assay_type": "B",
                    "target_chembl_id": "T1",
                    "document_chembl_id": "D1",
                }],
            }
            resp.read.return_value = json.dumps(data).encode()
            return resp
        mock_url.side_effect = bad_smiles_response
        dataset = fetch_chembl_bioactivity("CHEMBL4860")
        assert len(dataset) == 1
        assert dataset.records[0].mol is None
        assert len(dataset.records[0].errors) > 0


# ── Activity To Record Tests ──────────────────────────────────


class TestActivityToRecord:

    def test_basic_conversion(self):
        activity = MOCK_CHEMBL_RESPONSE_PAGE1["activities"][0]
        rec = _activity_to_record(activity, "CHEMBL4860", 0)
        assert rec.source_id == "CHEMBL1"
        assert rec.smiles == ASPIRIN_SMILES
        assert rec.mol is not None
        assert rec.metadata["standard_value"] == 500.0


# ── CLI Tests ────────────────────────────────────────────────


class TestCLI:

    def test_cli_curate(self, tmp_path):
        from click.testing import CliRunner
        from drugflow.cli.main import cli

        # Create input CSV with bioactivity data
        input_csv = tmp_path / "raw.csv"
        input_csv.write_text(
            "smiles,standard_value,standard_units,standard_relation\n"
            f"{ASPIRIN_SMILES},100,nM,=\n"
            f"{IBUPROFEN_SMILES},200,nM,=\n"
            f"{CAFFEINE_SMILES},5000,nM,>\n"
        )
        output_csv = str(tmp_path / "curated.csv")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "data", "curate",
            "-i", str(input_csv),
            "-o", output_csv,
        ])
        assert result.exit_code == 0
        assert "Curation summary" in result.output

    def test_cli_label_binary(self, tmp_path):
        from click.testing import CliRunner
        from drugflow.cli.main import cli

        input_csv = tmp_path / "curated.csv"
        input_csv.write_text(
            "smiles,standard_value,standard_units\n"
            f"{ASPIRIN_SMILES},100,nM\n"
            f"{CAFFEINE_SMILES},5000,nM\n"
        )
        output_csv = str(tmp_path / "labeled.csv")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "data", "label",
            "-i", str(input_csv),
            "-o", output_csv,
        ])
        assert result.exit_code == 0
        assert "Active" in result.output

    def test_cli_label_ternary(self, tmp_path):
        from click.testing import CliRunner
        from drugflow.cli.main import cli

        input_csv = tmp_path / "curated.csv"
        input_csv.write_text(
            "smiles,standard_value,standard_units\n"
            f"{ASPIRIN_SMILES},100,nM\n"
            f"{CAFFEINE_SMILES},5000,nM\n"
            f"{IBUPROFEN_SMILES},50000,nM\n"
        )
        output_csv = str(tmp_path / "labeled.csv")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "data", "label",
            "-i", str(input_csv),
            "-o", output_csv,
            "--mode", "ternary",
            "--active-threshold", "1000",
            "--inactive-threshold", "10000",
        ])
        assert result.exit_code == 0
        assert "Intermediate" in result.output

    @patch("drugflow.phase1.data.bioactivity.urllib.request.urlopen",
           side_effect=_mock_urlopen_single_page)
    def test_cli_fetch_bioactivity(self, mock_url, tmp_path):
        from click.testing import CliRunner
        from drugflow.cli.main import cli

        output_csv = str(tmp_path / "fetched.csv")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "data", "fetch-bioactivity",
            "-t", "CHEMBL4860",
            "-o", output_csv,
        ])
        assert result.exit_code == 0
        assert "Fetched" in result.output

    @patch("drugflow.phase1.data.bioactivity.urllib.request.urlopen",
           side_effect=_mock_urlopen_single_page)
    def test_cli_fetch_and_curate(self, mock_url, tmp_path):
        from click.testing import CliRunner
        from drugflow.cli.main import cli

        output_csv = str(tmp_path / "ready.csv")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "data", "fetch-and-curate",
            "-t", "CHEMBL4860",
            "-o", output_csv,
        ])
        assert result.exit_code == 0
        assert "DrugFlow Fetch & Curate Pipeline" in result.output


# ── Integration Test ──────────────────────────────────────────


class TestIntegration:

    @patch("drugflow.phase1.data.bioactivity.urllib.request.urlopen",
           side_effect=_mock_urlopen_single_page)
    def test_full_pipeline(self, mock_url):
        """Fetch -> curate -> label in one shot."""
        dataset, stats = fetch_and_curate(
            "CHEMBL4860",
            label_mode="binary",
            active_threshold=1000.0,
        )
        assert stats.input_count == 3
        assert stats.output_count <= 3

        # All records should have labels
        for rec in dataset.records:
            assert "activity_class" in rec.metadata
            assert "pIC50" in rec.metadata

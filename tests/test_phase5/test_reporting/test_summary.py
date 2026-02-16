"""Tests for pipeline summary and reporting."""

import pytest

from claudedd.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus
from claudedd.phase5.reporting.summary import (
    create_pipeline_report,
    summarize_dataset,
    summarize_generation,
    summarize_scoring,
)


class TestSummarizeDataset:
    """Tests for dataset summarization."""

    def test_basic_summary(self, sample_dataset):
        """Summarize a basic dataset."""
        result = summarize_dataset(sample_dataset)
        assert "total_records" in result
        assert "valid_records" in result
        assert "property_stats" in result
        assert "filter_stats" in result

    def test_summary_with_properties(self, computed_dataset):
        """Summary includes property statistics."""
        result = summarize_dataset(computed_dataset)
        stats = result["property_stats"]
        # computed_dataset has MolWt computed
        if "MolWt" in stats:
            assert "mean" in stats["MolWt"]
            assert "min" in stats["MolWt"]
            assert "max" in stats["MolWt"]

    def test_summary_counts(self, sample_dataset):
        """Counts match dataset."""
        result = summarize_dataset(sample_dataset)
        assert result["total_records"] == len(sample_dataset.records)


class TestSummarizeGeneration:
    """Tests for generation summary."""

    def test_generation_metrics(self, sample_dataset):
        """Compute generation metrics."""
        from rdkit import Chem
        # Create a small "generated" set
        gen_records = []
        for smi in ["c1ccccc1", "CCO", sample_dataset.valid_records[0].smiles]:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                rec = MoleculeRecord(
                    mol=mol, smiles=smi,
                    source_id=f"gen_{smi[:5]}", status=MoleculeStatus.RAW,
                )
                gen_records.append(rec)
        generated = MoleculeDataset(records=gen_records, name="gen")

        result = summarize_generation(sample_dataset, generated)
        assert "n_generated" in result
        assert "n_valid" in result
        assert "validity_rate" in result
        assert "novelty_rate" in result
        assert 0.0 <= result["validity_rate"] <= 1.0
        assert 0.0 <= result["novelty_rate"] <= 1.0


class TestSummarizeScoring:
    """Tests for scoring summary."""

    def test_scored_dataset(self, sample_dataset):
        """Summarize scored dataset."""
        # Add fake scores
        for i, rec in enumerate(sample_dataset.valid_records):
            rec.properties["composite_score"] = 0.5 + i * 0.1
        result = summarize_scoring(sample_dataset)
        assert "n_scored" in result
        assert result["n_scored"] > 0
        assert "score_stats" in result

    def test_unscored_dataset(self, sample_dataset):
        """Unscored dataset returns zeros."""
        result = summarize_scoring(sample_dataset)
        assert result["n_scored"] == 0


class TestCreatePipelineReport:
    """Tests for full pipeline report."""

    def test_report_structure(self):
        """Report has required fields."""
        report = create_pipeline_report(
            {"phase1": {"n_molecules": 10}},
            metadata={"workflow": "test"},
        )
        assert "claudedd_version" in report
        assert "timestamp" in report
        assert "stages" in report
        assert "metadata" in report
        assert report["metadata"]["workflow"] == "test"

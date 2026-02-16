"""Tests for project export utilities."""

import json
import os

import pytest

from claudedd.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus
from claudedd.phase5.reporting.export import (
    create_provenance_record,
    export_project_json,
    export_results_csv,
)


class TestExportProjectJSON:
    """Tests for JSON export."""

    def test_export_creates_file(self, tmp_path):
        """Export creates a valid JSON file."""
        report = {"workflow": "test", "stages": {"phase1": {"n": 10}}}
        out = str(tmp_path / "report.json")
        result = export_project_json(report, out)
        assert os.path.exists(result)

        with open(result) as f:
            data = json.load(f)
        assert data["workflow"] == "test"

    def test_export_nested_data(self, tmp_path):
        """Export handles nested structures."""
        report = {"stages": {"a": {"b": [1, 2, 3]}}}
        out = str(tmp_path / "nested.json")
        export_project_json(report, out)

        with open(out) as f:
            data = json.load(f)
        assert data["stages"]["a"]["b"] == [1, 2, 3]


class TestExportResultsCSV:
    """Tests for CSV export."""

    def test_export_creates_csv(self, sample_dataset, tmp_path):
        """Export creates a CSV file."""
        # Add a property
        for rec in sample_dataset.valid_records:
            rec.properties["MolWt"] = 180.0

        out = str(tmp_path / "results.csv")
        result = export_results_csv(sample_dataset, out)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_export_specific_columns(self, sample_dataset, tmp_path):
        """Export only specified columns."""
        for rec in sample_dataset.valid_records:
            rec.properties["MolWt"] = 180.0
            rec.properties["LogP"] = 1.5

        out = str(tmp_path / "specific.csv")
        export_results_csv(sample_dataset, out, columns=["MolWt"])

        with open(out) as f:
            header = f.readline().strip()
        assert "MolWt" in header

    def test_export_empty_dataset(self, tmp_path):
        """Empty dataset creates file with header."""
        empty = MoleculeDataset(records=[], name="empty")
        out = str(tmp_path / "empty.csv")
        export_results_csv(empty, out)
        assert os.path.exists(out)


class TestCreateProvenanceRecord:
    """Tests for provenance tracking."""

    def test_provenance_basic(self, sample_dataset):
        """Provenance record has required fields."""
        # Add some provenance
        for rec in sample_dataset.valid_records:
            rec.add_provenance("validate")
            rec.add_provenance("properties")

        result = create_provenance_record(sample_dataset)
        assert "timestamp" in result
        assert "dataset_name" in result
        assert "n_records" in result
        assert "phases_applied" in result
        assert "records" in result

    def test_provenance_tracks_phases(self, sample_dataset):
        """Provenance tracks which phases were applied."""
        for rec in sample_dataset.valid_records:
            rec.add_provenance("validate")
            rec.add_provenance("fingerprint")
            rec.add_provenance("brics_gen")

        result = create_provenance_record(sample_dataset)
        phases = result["phases_applied"]
        assert isinstance(phases, list)

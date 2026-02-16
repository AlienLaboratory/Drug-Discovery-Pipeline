"""Tests for lead optimization workflow."""

import json
import os

import pytest

from claudedd.phase5.workflows.lead_optimization import run_lead_optimization


ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"


class TestLeadOptimization:
    """Tests for lead optimization pipeline."""

    def test_basic_run(self, tmp_path):
        """Basic lead optimization runs end-to-end."""
        output_dir = str(tmp_path / "optimize")
        report = run_lead_optimization(
            ASPIRIN_SMILES, output_dir,
            n_analogs=10, top_n=5, seed=42,
        )

        assert isinstance(report, dict)
        assert "stages" in report
        assert "lead" in report["stages"]
        assert report["stages"]["lead"]["smiles"] == ASPIRIN_SMILES

    def test_output_files(self, tmp_path):
        """Expected output files are created."""
        output_dir = str(tmp_path / "optimize_files")
        run_lead_optimization(
            ASPIRIN_SMILES, output_dir,
            n_analogs=10, top_n=3, seed=42,
        )

        assert os.path.exists(os.path.join(output_dir, "optimized_candidates.csv"))
        assert os.path.exists(os.path.join(output_dir, "all_analogs.csv"))
        assert os.path.exists(os.path.join(output_dir, "report_summary.json"))

    def test_report_metadata(self, tmp_path):
        """Report contains correct metadata."""
        output_dir = str(tmp_path / "optimize_meta")
        run_lead_optimization(
            ASPIRIN_SMILES, output_dir,
            n_analogs=10, top_n=3, seed=42,
        )

        with open(os.path.join(output_dir, "report_summary.json")) as f:
            data = json.load(f)
        assert data["metadata"]["workflow"] == "lead_optimization"
        assert data["metadata"]["lead_smiles"] == ASPIRIN_SMILES

    def test_invalid_smiles_raises(self, tmp_path):
        """Invalid SMILES raises ValueError."""
        with pytest.raises(ValueError, match="Invalid lead"):
            run_lead_optimization(
                "INVALID_SMILES", str(tmp_path / "fail"),
            )

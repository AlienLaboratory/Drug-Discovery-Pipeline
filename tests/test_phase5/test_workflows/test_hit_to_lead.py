"""Tests for hit-to-lead workflow."""

import json
import os

import pytest

from claudedd.phase5.workflows.hit_to_lead import run_hit_to_lead


class TestHitToLead:
    """Tests for hit-to-lead pipeline."""

    def test_basic_run(self, sample_csv_path, tmp_path):
        """Basic hit-to-lead pipeline runs end-to-end."""
        output_dir = str(tmp_path / "hit2lead")
        report = run_hit_to_lead(
            sample_csv_path, output_dir,
            n_generate=10, top_n=5, seed=42,
        )

        assert isinstance(report, dict)
        assert "stages" in report
        assert "phase1_profile" in report["stages"]
        assert "phase2_scoring" in report["stages"]

    def test_output_files_created(self, sample_csv_path, tmp_path):
        """Pipeline creates expected output files."""
        output_dir = str(tmp_path / "hit2lead_files")
        run_hit_to_lead(
            sample_csv_path, output_dir,
            n_generate=5, top_n=3, seed=42,
        )

        assert os.path.exists(os.path.join(output_dir, "ranked_candidates.csv"))
        assert os.path.exists(os.path.join(output_dir, "all_results.csv"))
        assert os.path.exists(os.path.join(output_dir, "report_summary.json"))

    def test_report_json_valid(self, sample_csv_path, tmp_path):
        """Report JSON is valid and contains metadata."""
        output_dir = str(tmp_path / "hit2lead_json")
        run_hit_to_lead(
            sample_csv_path, output_dir,
            n_generate=5, top_n=3, seed=42,
        )

        with open(os.path.join(output_dir, "report_summary.json")) as f:
            data = json.load(f)
        assert data["metadata"]["workflow"] == "hit_to_lead"
        assert "claudedd_version" in data

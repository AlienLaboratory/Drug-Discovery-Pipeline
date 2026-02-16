"""Tests for de novo design workflow."""

import json
import os

import pytest

from drugflow.phase5.workflows.denovo_design import run_denovo_design


class TestDeNovoDesign:
    """Tests for de novo design pipeline."""

    def test_brics_strategy(self, sample_csv_path, tmp_path):
        """BRICS strategy runs end-to-end."""
        output_dir = str(tmp_path / "denovo_brics")
        report = run_denovo_design(
            sample_csv_path, output_dir,
            strategy="brics", n_generate=20, top_n=5, seed=42,
        )

        assert isinstance(report, dict)
        assert "stages" in report
        assert "phase3_generation" in report["stages"]

    def test_mutate_strategy(self, sample_csv_path, tmp_path):
        """Mutation strategy runs end-to-end."""
        output_dir = str(tmp_path / "denovo_mutate")
        report = run_denovo_design(
            sample_csv_path, output_dir,
            strategy="mutate", n_generate=10, top_n=3, seed=42,
        )

        assert "phase3_generation" in report["stages"]
        gen = report["stages"]["phase3_generation"]
        assert gen["n_generated"] > 0

    def test_output_files(self, sample_csv_path, tmp_path):
        """Expected output files are created."""
        output_dir = str(tmp_path / "denovo_files")
        run_denovo_design(
            sample_csv_path, output_dir,
            strategy="mutate", n_generate=10, top_n=3, seed=42,
        )

        assert os.path.exists(os.path.join(output_dir, "generated_scored.csv"))
        assert os.path.exists(os.path.join(output_dir, "all_generated.csv"))
        assert os.path.exists(os.path.join(output_dir, "report_summary.json"))

    def test_invalid_strategy_raises(self, sample_csv_path, tmp_path):
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            run_denovo_design(
                sample_csv_path, str(tmp_path / "fail"),
                strategy="invalid",
            )

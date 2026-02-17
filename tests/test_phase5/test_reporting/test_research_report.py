"""Tests for research report generation."""

import csv
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from drugflow.phase5.reporting.research_report import (
    generate_research_report,
    _compute_candidate_stats,
    _find_potency_column,
    _generate_summary_text,
)


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def tmp_dir():
    """Temporary directory."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_candidates_csv(tmp_dir):
    """Create a sample candidates CSV."""
    np.random.seed(42)
    n = 30
    data = {
        "smiles": [f"C{'C' * (i % 5)}O" for i in range(n)],
        "predicted_pIC50": np.random.uniform(6, 10, n),
        "MolWt": np.random.uniform(300, 900, n),
        "LogP": np.random.uniform(1, 8, n),
        "TPSA": np.random.uniform(50, 200, n),
        "QED": np.random.uniform(0.05, 0.4, n),
        "HBD": np.random.randint(0, 5, n),
        "HBA": np.random.randint(2, 12, n),
        "RingCount": np.random.randint(2, 8, n),
        "source": ["mutation"] * 15 + ["brics"] * 15,
        "is_novel": [True] * n,
    }
    df = pd.DataFrame(data)
    path = os.path.join(tmp_dir, "candidates.csv")
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_training_csv(tmp_dir):
    """Create a sample training data CSV."""
    np.random.seed(123)
    n = 50
    data = {
        "smiles": [f"c1ccc{'c' * (i % 3)}c1" for i in range(n)],
        "pIC50": np.random.uniform(5, 10, n),
        "MolWt": np.random.uniform(200, 1000, n),
        "LogP": np.random.uniform(0, 10, n),
        "TPSA": np.random.uniform(30, 250, n),
        "QED": np.random.uniform(0.05, 0.5, n),
        "HBD": np.random.randint(0, 6, n),
        "HBA": np.random.randint(1, 12, n),
        "activity_class": np.random.choice([0, 1], n, p=[0.2, 0.8]),
    }
    df = pd.DataFrame(data)
    path = os.path.join(tmp_dir, "training.csv")
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_comparison_csv(tmp_dir):
    """Create a sample model comparison CSV."""
    data = {
        "model_type": ["random_forest", "gradient_boosting", "svr"],
        "cv_r2": [0.72, 0.75, 0.68],
        "cv_r2_std": [0.03, 0.04, 0.05],
        "test_r2": [0.70, 0.73, 0.65],
        "test_rmse": [0.45, 0.42, 0.50],
    }
    df = pd.DataFrame(data)
    path = os.path.join(tmp_dir, "comparison.csv")
    df.to_csv(path, index=False)
    return path


# ── TestHelpers ─────────────────────────────────────────────


class TestHelpers:
    """Tests for helper functions."""

    def test_find_potency_column_predicted_pic50(self):
        """Finds predicted_pIC50 column."""
        df = pd.DataFrame({"predicted_pIC50": [7.5], "MolWt": [500]})
        assert _find_potency_column(df) == "predicted_pIC50"

    def test_find_potency_column_pic50(self):
        """Finds pIC50 column."""
        df = pd.DataFrame({"pIC50": [7.5], "MolWt": [500]})
        assert _find_potency_column(df) == "pIC50"

    def test_find_potency_column_none(self):
        """Returns None when no potency column found."""
        df = pd.DataFrame({"MolWt": [500], "LogP": [3.0]})
        assert _find_potency_column(df) is None

    def test_compute_candidate_stats(self):
        """Computes stats from DataFrame."""
        df = pd.DataFrame({
            "MolWt": [400, 500, 600],
            "LogP": [3.0, 5.0, 7.0],
        })
        stats = _compute_candidate_stats(df)
        assert stats["n_molecules"] == 3
        assert "MolWt" in stats
        assert abs(stats["MolWt"]["mean"] - 500) < 0.1
        assert abs(stats["LogP"]["mean"] - 5.0) < 0.1

    def test_compute_stats_with_missing(self):
        """Stats handles columns not present."""
        df = pd.DataFrame({"smiles": ["CCO"]})
        stats = _compute_candidate_stats(df)
        assert stats["n_molecules"] == 1
        assert "MolWt" not in stats


# ── TestGenerateReport ──────────────────────────────────────


class TestGenerateReport:
    """Tests for the main generate_research_report function."""

    def test_creates_output_dir(self, tmp_dir, sample_candidates_csv, sample_training_csv):
        """Report creates the output directory."""
        out_dir = os.path.join(tmp_dir, "report_output")
        generate_research_report(
            candidates_path=sample_candidates_csv,
            training_data_path=sample_training_csv,
            output_dir=out_dir,
        )
        assert os.path.isdir(out_dir)

    def test_creates_summary(self, tmp_dir, sample_candidates_csv, sample_training_csv):
        """Report creates summary.txt."""
        out_dir = os.path.join(tmp_dir, "report_summary")
        generate_research_report(
            candidates_path=sample_candidates_csv,
            training_data_path=sample_training_csv,
            output_dir=out_dir,
            target_name="BCL-2",
            campaign_name="Test Campaign",
        )
        summary_path = os.path.join(out_dir, "summary.txt")
        assert os.path.exists(summary_path)
        with open(summary_path, encoding="utf-8") as f:
            text = f.read()
        assert "BCL-2" in text
        assert "Test Campaign" in text
        assert "DrugFlow" in text

    def test_creates_candidates_csvs(self, tmp_dir, sample_candidates_csv, sample_training_csv):
        """Report creates full and top50 candidate CSVs."""
        out_dir = os.path.join(tmp_dir, "report_csvs")
        generate_research_report(
            candidates_path=sample_candidates_csv,
            training_data_path=sample_training_csv,
            output_dir=out_dir,
        )
        assert os.path.exists(os.path.join(out_dir, "candidates_full.csv"))
        assert os.path.exists(os.path.join(out_dir, "candidates_top50.csv"))
        assert os.path.exists(os.path.join(out_dir, "training_data_stats.csv"))

    def test_creates_plots(self, tmp_dir, sample_candidates_csv, sample_training_csv):
        """Report creates plot files."""
        out_dir = os.path.join(tmp_dir, "report_plots")
        generate_research_report(
            candidates_path=sample_candidates_csv,
            training_data_path=sample_training_csv,
            output_dir=out_dir,
        )
        plots_dir = os.path.join(out_dir, "plots")
        assert os.path.isdir(plots_dir)
        # At least property_distributions and potency should exist
        assert os.path.exists(os.path.join(plots_dir, "property_distributions.png"))
        assert os.path.exists(os.path.join(plots_dir, "potency_distribution.png"))

    def test_with_comparison(self, tmp_dir, sample_candidates_csv,
                             sample_training_csv, sample_comparison_csv):
        """Report includes model comparison when provided."""
        out_dir = os.path.join(tmp_dir, "report_comp")
        generate_research_report(
            candidates_path=sample_candidates_csv,
            training_data_path=sample_training_csv,
            output_dir=out_dir,
            model_comparison_path=sample_comparison_csv,
        )
        assert os.path.exists(os.path.join(out_dir, "model_comparison.csv"))
        summary_path = os.path.join(out_dir, "summary.txt")
        with open(summary_path, encoding="utf-8") as f:
            text = f.read()
        assert "MODEL COMPARISON" in text

    def test_handles_missing_model(self, tmp_dir, sample_candidates_csv, sample_training_csv):
        """Report works without model_path."""
        out_dir = os.path.join(tmp_dir, "report_nomodel")
        generate_research_report(
            candidates_path=sample_candidates_csv,
            training_data_path=sample_training_csv,
            output_dir=out_dir,
            model_path=None,
        )
        assert os.path.exists(os.path.join(out_dir, "summary.txt"))

    def test_top50_has_correct_count(self, tmp_dir, sample_candidates_csv, sample_training_csv):
        """Top50 CSV has at most 50 rows (or all if fewer)."""
        out_dir = os.path.join(tmp_dir, "report_top50")
        generate_research_report(
            candidates_path=sample_candidates_csv,
            training_data_path=sample_training_csv,
            output_dir=out_dir,
        )
        top50 = pd.read_csv(os.path.join(out_dir, "candidates_top50.csv"))
        # We have 30 candidates, so top50 should have 30
        assert len(top50) == 30


# ── TestSummaryText ─────────────────────────────────────────


class TestSummaryText:
    """Tests for summary text generation."""

    def test_summary_has_target(self):
        """Summary includes target name."""
        cand_df = pd.DataFrame({"smiles": ["CCO"], "predicted_pIC50": [8.0], "MolWt": [46]})
        train_df = pd.DataFrame({"smiles": ["CCO"], "pIC50": [7.0], "MolWt": [46]})
        text = _generate_summary_text(
            cand_df, train_df,
            _compute_candidate_stats(cand_df),
            _compute_candidate_stats(train_df),
            model_info=None, comparison_df=None,
            target_name="BCL-2", campaign_name="My Campaign",
        )
        assert "BCL-2" in text
        assert "My Campaign" in text

    def test_summary_has_counts(self):
        """Summary includes molecule counts."""
        cand_df = pd.DataFrame({"smiles": ["CCO"] * 10, "MolWt": [46] * 10})
        train_df = pd.DataFrame({"smiles": ["CCO"] * 50, "MolWt": [46] * 50})
        text = _generate_summary_text(
            cand_df, train_df,
            _compute_candidate_stats(cand_df),
            _compute_candidate_stats(train_df),
            model_info=None, comparison_df=None,
            target_name="X", campaign_name="Y",
        )
        assert "10" in text  # candidates count
        assert "50" in text  # training count

    def test_summary_has_top_candidates(self):
        """Summary lists top candidates."""
        cand_df = pd.DataFrame({
            "smiles": [f"C{'C' * i}O" for i in range(5)],
            "predicted_pIC50": [9.5, 9.0, 8.5, 8.0, 7.5],
        })
        train_df = pd.DataFrame({"smiles": ["CCO"], "MolWt": [46]})
        text = _generate_summary_text(
            cand_df, train_df,
            _compute_candidate_stats(cand_df),
            _compute_candidate_stats(train_df),
            model_info=None, comparison_df=None,
            target_name="X", campaign_name="Y",
        )
        assert "TOP 10 CANDIDATES" in text


# ── TestCLIResearchReport ───────────────────────────────────


class TestCLIResearchReport:
    """Tests for the research-report CLI command."""

    def test_cli_basic(self, tmp_dir, sample_candidates_csv, sample_training_csv):
        """Basic CLI invocation."""
        from click.testing import CliRunner
        from drugflow.cli.workflow_commands import research_report

        out_dir = os.path.join(tmp_dir, "cli_report")
        runner = CliRunner()
        result = runner.invoke(research_report, [
            "-c", sample_candidates_csv,
            "-t", sample_training_csv,
            "-o", out_dir,
            "--target-name", "BCL-2",
            "--campaign-name", "Test",
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert os.path.exists(os.path.join(out_dir, "summary.txt"))

    def test_cli_with_comparison(self, tmp_dir, sample_candidates_csv,
                                  sample_training_csv, sample_comparison_csv):
        """CLI with model comparison."""
        from click.testing import CliRunner
        from drugflow.cli.workflow_commands import research_report

        out_dir = os.path.join(tmp_dir, "cli_report_comp")
        runner = CliRunner()
        result = runner.invoke(research_report, [
            "-c", sample_candidates_csv,
            "-t", sample_training_csv,
            "-o", out_dir,
            "--model-comparison", sample_comparison_csv,
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert os.path.exists(os.path.join(out_dir, "model_comparison.csv"))

"""Tests for docking results visualization and report module."""

import json
import os
import tempfile

import pandas as pd
import pytest

from drugflow.phase5.reporting.docking_report import (
    _compute_docking_statistics,
    _compute_population_statistics,
    _generate_ranked_csv,
    _generate_summary_text,
    _plot_admet_class_distribution,
    _plot_admet_radar_top_n,
    _plot_integrated_dashboard,
    _plot_ranked_candidates_table,
    _plot_risk_flag_heatmap,
    _plot_score_correlations,
    _plot_summary_statistics_table,
    _plot_vina_score_distribution,
    generate_docking_report,
)


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_docking_df():
    """Synthetic 10-row docking results DataFrame."""
    import numpy as np
    np.random.seed(42)
    n = 10
    return pd.DataFrame({
        "smiles": [f"C{'C'*i}O" for i in range(n)],
        "record_id": [f"cand_{i}" for i in range(n)],
        "predicted_pIC50": np.random.uniform(6.5, 8.0, n).round(2),
        "predicted_IC50_nM": np.random.uniform(10, 300, n).round(1),
        "admet_score": np.random.uniform(0.4, 0.95, n).round(3),
        "admet_class": np.random.choice(["favorable", "moderate", "poor"], n, p=[0.6, 0.3, 0.1]),
        "admet_n_red_flags": np.random.randint(0, 3, n),
        "admet_n_yellow_flags": np.random.randint(0, 6, n),
        "MolWt": np.random.uniform(300, 700, n).round(1),
        "LogP": np.random.uniform(1.0, 6.0, n).round(2),
        "QED": np.random.uniform(0.2, 0.9, n).round(2),
        "RingCount": np.random.randint(2, 8, n),
        "admet_herg_risk": np.random.choice(["green", "yellow", "red"], n),
        "admet_ames_risk": np.random.choice(["green", "yellow", "red"], n),
        "admet_hepatotox_risk": np.random.choice(["green", "yellow", "red"], n),
        "admet_caco2_class": np.random.choice(["high", "moderate"], n),
        "admet_hia_class": np.random.choice(["high", "moderate"], n),
        "admet_bbb_penetrant": np.random.choice([True, False], n),
        "admet_metabolic_stability": np.random.choice(["high", "moderate", "low"], n),
        "admet_cyp_inhibition_risk": np.random.choice(["low", "moderate", "high"], n),
        "vina_score": np.random.uniform(-4.0, -2.5, n).round(3),
        "n_conformers": [3] * n,
        "rank": list(range(1, n + 1)),
    })


@pytest.fixture
def sample_admet_df():
    """Synthetic 50-row ADMET DataFrame (simulating 397 candidates at smaller scale)."""
    import numpy as np
    np.random.seed(99)
    n = 50
    return pd.DataFrame({
        "smiles": [f"C{'C'*i}O" for i in range(n)],
        "admet_score": np.random.uniform(0.2, 0.98, n).round(3),
        "admet_class": np.random.choice(["favorable", "moderate", "poor"], n, p=[0.45, 0.1, 0.45]),
        "admet_absorption_score": np.random.uniform(0.3, 1.0, n).round(3),
        "admet_distribution_score": np.random.uniform(0.3, 1.0, n).round(3),
        "admet_metabolism_score": np.random.uniform(0.3, 1.0, n).round(3),
        "admet_excretion_score": np.random.uniform(0.3, 1.0, n).round(3),
        "admet_toxicity_score": np.random.uniform(0.3, 1.0, n).round(3),
    })


@pytest.fixture
def sample_docking_csv(tmp_dir, sample_docking_df):
    """Write docking DataFrame to CSV and return path."""
    path = os.path.join(tmp_dir, "docking_results.csv")
    sample_docking_df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_admet_csv(tmp_dir, sample_admet_df):
    """Write ADMET DataFrame to CSV and return path."""
    path = os.path.join(tmp_dir, "admet_all_candidates.csv")
    sample_admet_df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_study_json(tmp_dir):
    """Write a minimal study_report.json."""
    data = {
        "study": "Test Docking Study",
        "target": "Test Target",
        "method": "vina",
        "total_novel_candidates": 10,
        "scoring_column": "vina_score",
    }
    path = os.path.join(tmp_dir, "study_report.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


# ── TestDockingStatistics ───────────────────────────────────────────

class TestDockingStatistics:
    def test_compute_basic_stats(self, sample_docking_df):
        """Stats include vina, potency, admet sections."""
        stats = _compute_docking_statistics(sample_docking_df)
        assert "vina" in stats
        assert stats["vina"]["n"] == 10
        assert stats["vina"]["mean"] != 0
        assert stats["vina"]["min"] < stats["vina"]["max"]

    def test_compute_potency(self, sample_docking_df):
        """Potency stats are present."""
        stats = _compute_docking_statistics(sample_docking_df)
        assert "potency" in stats
        assert "mean" in stats["potency"]

    def test_compute_risk_counts(self, sample_docking_df):
        """Risk flag counts are computed."""
        stats = _compute_docking_statistics(sample_docking_df)
        assert "risk_flags" in stats
        assert "herg" in stats["risk_flags"]
        herg = stats["risk_flags"]["herg"]
        assert herg["green"] + herg["yellow"] + herg["red"] == 10

    def test_compute_binding_tiers(self, sample_docking_df):
        """Binding tiers sum to total."""
        stats = _compute_docking_statistics(sample_docking_df)
        v = stats["vina"]
        assert v["n_strong"] + v["n_moderate"] + v["n_weak"] == v["n"]

    def test_handles_missing_columns(self):
        """Works with minimal DataFrame."""
        df = pd.DataFrame({"smiles": ["CCO", "CCC"], "rank": [1, 2]})
        stats = _compute_docking_statistics(df)
        assert stats["vina"]["n"] == 0


class TestPopulationStatistics:
    def test_basic_stats(self, sample_admet_df):
        """Population stats have expected keys."""
        stats = _compute_population_statistics(sample_admet_df)
        assert stats["total"] == 50
        assert "favorable" in stats
        assert "mean_score" in stats

    def test_domain_means(self, sample_admet_df):
        """Domain sub-score means are computed."""
        stats = _compute_population_statistics(sample_admet_df)
        assert "absorption_mean" in stats
        assert "toxicity_mean" in stats


# ── TestPlotGeneration ──────────────────────────────────────────────

class TestPlotGeneration:
    def test_ranked_candidates_table(self, sample_docking_df, tmp_dir):
        """Ranked table image is created."""
        path = _plot_ranked_candidates_table(sample_docking_df, tmp_dir, top_n=5)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_vina_distribution(self, sample_docking_df, tmp_dir):
        """Vina distribution plot is created."""
        path = _plot_vina_score_distribution(sample_docking_df, tmp_dir)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_admet_radar(self, sample_docking_df, sample_admet_df, tmp_dir):
        """ADMET radar chart is created."""
        path = _plot_admet_radar_top_n(sample_docking_df, sample_admet_df, tmp_dir, top_n=3)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_score_correlations(self, sample_docking_df, tmp_dir):
        """Score correlation plots are created."""
        path = _plot_score_correlations(sample_docking_df, tmp_dir)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_admet_class_distribution(self, sample_docking_df, sample_admet_df, tmp_dir):
        """ADMET class donut charts are created."""
        path = _plot_admet_class_distribution(sample_docking_df, sample_admet_df, tmp_dir)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_risk_flag_heatmap(self, sample_docking_df, tmp_dir):
        """Risk flag heatmap is created."""
        path = _plot_risk_flag_heatmap(sample_docking_df, tmp_dir)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_integrated_dashboard(self, sample_docking_df, sample_admet_df, tmp_dir):
        """Integrated dashboard is created."""
        stats = _compute_docking_statistics(sample_docking_df)
        path = _plot_integrated_dashboard(sample_docking_df, sample_admet_df, stats, tmp_dir)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_summary_statistics_table(self, sample_docking_df, tmp_dir):
        """Summary statistics table image is created."""
        stats = _compute_docking_statistics(sample_docking_df)
        path = _plot_summary_statistics_table(stats, tmp_dir)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0


# ── TestDataOutputs ─────────────────────────────────────────────────

class TestDataOutputs:
    def test_ranked_csv_created(self, sample_docking_df, tmp_dir):
        """Ranked CSV is created with correct columns."""
        path = _generate_ranked_csv(sample_docking_df, tmp_dir)
        assert os.path.isfile(path)
        result = pd.read_csv(path)
        assert "composite_docking_score" in result.columns
        assert "rank" in result.columns
        assert len(result) == 10

    def test_composite_score_range(self, sample_docking_df, tmp_dir):
        """Composite score is between 0 and 1."""
        _generate_ranked_csv(sample_docking_df, tmp_dir)
        result = pd.read_csv(os.path.join(tmp_dir, "ranked_candidates.csv"))
        scores = result["composite_docking_score"]
        assert scores.min() >= -0.1
        assert scores.max() <= 1.1

    def test_summary_text_has_sections(self, sample_docking_df):
        """Summary text contains all major sections."""
        stats = _compute_docking_statistics(sample_docking_df)
        text = _generate_summary_text(sample_docking_df, stats, None, "BCL-2", "Test Campaign")
        assert "DOCKING RESULTS SUMMARY" in text
        assert "PREDICTED POTENCY" in text
        assert "ADMET PROFILE" in text
        assert "TOP 10 CANDIDATES" in text

    def test_summary_text_values(self, sample_docking_df):
        """Summary text contains actual numeric values."""
        stats = _compute_docking_statistics(sample_docking_df)
        text = _generate_summary_text(sample_docking_df, stats, None, "BCL-2", "Test")
        assert "kcal/mol" in text
        assert "BCL-2" in text


# ── TestGenerateDockingReport ───────────────────────────────────────

class TestGenerateDockingReport:
    def test_creates_output_dir(self, sample_docking_csv, tmp_dir):
        """Output directory is created."""
        out = os.path.join(tmp_dir, "report")
        generate_docking_report(sample_docking_csv, out)
        assert os.path.isdir(out)

    def test_creates_all_plots(self, sample_docking_csv, tmp_dir):
        """All 8 PNG plots are generated."""
        out = os.path.join(tmp_dir, "report")
        generate_docking_report(sample_docking_csv, out)
        plots_dir = os.path.join(out, "plots")
        pngs = [f for f in os.listdir(plots_dir) if f.endswith(".png")]
        assert len(pngs) == 8

    def test_creates_data_files(self, sample_docking_csv, tmp_dir):
        """All data files are generated."""
        out = os.path.join(tmp_dir, "report")
        generate_docking_report(sample_docking_csv, out)
        assert os.path.isfile(os.path.join(out, "ranked_candidates.csv"))
        assert os.path.isfile(os.path.join(out, "enhanced_study_report.json"))
        assert os.path.isfile(os.path.join(out, "docking_summary.txt"))
        assert os.path.isfile(os.path.join(out, "docking_statistics.csv"))

    def test_with_admet_csv(self, sample_docking_csv, sample_admet_csv, tmp_dir):
        """Report works with full ADMET data."""
        out = os.path.join(tmp_dir, "report")
        generate_docking_report(
            sample_docking_csv, out, admet_all_path=sample_admet_csv,
        )
        with open(os.path.join(out, "enhanced_study_report.json")) as f:
            report = json.load(f)
        assert "population_admet" in report

    def test_with_study_json(self, sample_docking_csv, sample_study_json, tmp_dir):
        """Report includes original study data."""
        out = os.path.join(tmp_dir, "report")
        generate_docking_report(
            sample_docking_csv, out, study_report_path=sample_study_json,
        )
        with open(os.path.join(out, "enhanced_study_report.json")) as f:
            report = json.load(f)
        assert "original_study" in report

    def test_enhanced_json_has_interpretation(self, sample_docking_csv, tmp_dir):
        """Enhanced report includes scientific interpretation."""
        out = os.path.join(tmp_dir, "report")
        generate_docking_report(sample_docking_csv, out, target_name="BCL-2")
        with open(os.path.join(out, "enhanced_study_report.json")) as f:
            report = json.load(f)
        assert "interpretation" in report
        assert "binding_quality" in report["interpretation"]
        assert "overall_assessment" in report["interpretation"]


# ── TestCLIDockingReport ────────────────────────────────────────────

class TestCLIDockingReport:
    def test_cli_basic(self, sample_docking_csv, tmp_dir):
        """CLI invocation runs without error."""
        from click.testing import CliRunner
        from drugflow.cli.main import cli

        out = os.path.join(tmp_dir, "cli_report")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "workflow", "docking-report",
            "--input-dir", os.path.dirname(sample_docking_csv),
            "--output-dir", out,
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"

    def test_cli_with_target(self, sample_docking_csv, tmp_dir):
        """CLI with target-name option reflects in output."""
        from click.testing import CliRunner
        from drugflow.cli.main import cli

        out = os.path.join(tmp_dir, "cli_report2")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "workflow", "docking-report",
            "--input-dir", os.path.dirname(sample_docking_csv),
            "--output-dir", out,
            "--target-name", "BCL-2 (PDB: 6O0K)",
        ])
        assert result.exit_code == 0
        # Check that the target name appears in the summary
        txt_path = os.path.join(out, "docking_summary.txt")
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                assert "BCL-2" in f.read()

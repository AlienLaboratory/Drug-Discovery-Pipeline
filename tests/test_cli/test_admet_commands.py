"""Tests for ADMET CLI commands."""

import os
import pytest
from click.testing import CliRunner
from rdkit import Chem

from drugflow.cli.main import cli


@pytest.fixture
def sample_csv(tmp_path):
    """Create a small CSV for testing."""
    csv_path = str(tmp_path / "test_mols.csv")
    with open(csv_path, "w") as f:
        f.write("smiles,name\n")
        f.write("CC(=O)Oc1ccccc1C(=O)O,aspirin\n")
        f.write("Cn1c(=O)c2c(ncn2C)n(C)c1=O,caffeine\n")
        f.write("c1ccccc1,benzene\n")
    return csv_path


class TestAdmetPredict:
    def test_basic_invocation(self, sample_csv, tmp_path):
        """drugflow admet predict runs without error."""
        output = str(tmp_path / "admet_out.csv")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "admet", "predict",
            "-i", sample_csv,
            "-o", output,
        ])
        assert result.exit_code == 0, f"CLI error: {result.output}"
        assert os.path.exists(output)

    def test_output_contains_scores(self, sample_csv, tmp_path):
        """Output mentions ADMET Score."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "admet", "predict",
            "-i", sample_csv,
        ])
        assert result.exit_code == 0
        assert "ADMET Score" in result.output

    def test_summary_command(self, sample_csv):
        """drugflow admet summary runs without error."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "admet", "summary",
            "-i", sample_csv,
        ])
        assert result.exit_code == 0
        assert "Domain scores" in result.output

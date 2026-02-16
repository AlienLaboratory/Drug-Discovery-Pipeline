"""Tests for CLI commands."""

from click.testing import CliRunner

from claudedd.cli.main import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "ClaudeDD" in result.output


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "1.0.0" in result.output


def test_data_load_csv(sample_csv_path):
    runner = CliRunner()
    result = runner.invoke(cli, [
        "data", "load", "--input", sample_csv_path,
    ])
    assert result.exit_code == 0
    assert "Loaded" in result.output


def test_data_info(sample_csv_path):
    runner = CliRunner()
    result = runner.invoke(cli, [
        "data", "info", "--input", sample_csv_path,
    ])
    assert result.exit_code == 0
    assert "Total molecules" in result.output


def test_analyze_properties(sample_csv_path, tmp_path):
    runner = CliRunner()
    output = str(tmp_path / "props.csv")
    result = runner.invoke(cli, [
        "analyze", "properties",
        "--input", sample_csv_path,
        "--output", output,
    ])
    assert result.exit_code == 0
    assert "MolWt" in result.output


def test_analyze_filter(sample_csv_path):
    runner = CliRunner()
    result = runner.invoke(cli, [
        "analyze", "filter",
        "--input", sample_csv_path,
    ])
    assert result.exit_code == 0
    assert "LIPINSKI" in result.output


def test_pipeline_init_config(tmp_path):
    runner = CliRunner()
    output = str(tmp_path / "config.yaml")
    result = runner.invoke(cli, [
        "pipeline", "init-config", "--output", output,
    ])
    assert result.exit_code == 0
    assert "Template config" in result.output

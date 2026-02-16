"""Tests for data loaders."""

import pytest

from claudedd.core.exceptions import FileFormatError
from claudedd.phase1.data.loaders import load_file, load_csv, load_smiles, load_sdf


def test_load_csv_basic(sample_csv_path):
    dataset = load_csv(sample_csv_path, smiles_column="smiles", id_column="id")
    assert len(dataset) == 3
    assert dataset[0].smiles == "CC(=O)Oc1ccccc1C(=O)O"


def test_load_csv_extra_columns(sample_csv_path):
    dataset = load_csv(sample_csv_path, smiles_column="smiles", id_column="id")
    assert "activity" in dataset[0].metadata


def test_load_smiles_basic(sample_smi_path):
    dataset = load_smiles(sample_smi_path)
    assert len(dataset) == 3
    assert dataset[0].source_id == "aspirin"


def test_load_sdf_basic(sample_sdf_path):
    dataset = load_sdf(sample_sdf_path)
    assert len(dataset) == 3
    assert all(r.mol is not None for r in dataset.records)


def test_load_file_auto_detect(sample_csv_path):
    dataset = load_file(sample_csv_path, format="auto", smiles_column="smiles")
    assert len(dataset) == 3


def test_load_file_nonexistent():
    with pytest.raises(FileNotFoundError):
        load_file("/nonexistent/file.csv")


def test_load_file_unsupported_format(tmp_path):
    p = tmp_path / "test.xyz"
    p.write_text("dummy")
    with pytest.raises(FileFormatError):
        load_file(str(p))


def test_load_csv_limit(sample_csv_path):
    dataset = load_csv(sample_csv_path, smiles_column="smiles", limit=2)
    assert len(dataset) == 2


def test_load_smiles_limit(sample_smi_path):
    dataset = load_smiles(sample_smi_path, limit=1)
    assert len(dataset) == 1

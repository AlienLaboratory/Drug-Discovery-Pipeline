"""Tests for core data models."""

from rdkit import Chem

from claudedd.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus


def test_molecule_record_creation(aspirin_mol):
    rec = MoleculeRecord(mol=aspirin_mol, smiles="CC(=O)Oc1ccccc1C(=O)O")
    assert rec.is_valid
    assert rec.status == MoleculeStatus.RAW
    assert rec.record_id  # auto-generated


def test_molecule_record_invalid_mol():
    rec = MoleculeRecord(mol=None)
    rec.add_error("parse failed")
    assert not rec.is_valid
    assert rec.status == MoleculeStatus.FAILED


def test_molecule_record_canonical_smiles(aspirin_mol):
    rec = MoleculeRecord(mol=aspirin_mol)
    smi = rec.canonical_smiles
    assert smi  # non-empty
    assert "C" in smi


def test_molecule_record_properties(aspirin_record):
    aspirin_record.set_property("MolWt", 180.16)
    assert aspirin_record.get_property("MolWt") == 180.16
    assert aspirin_record.get_property("nonexistent") is None
    assert aspirin_record.get_property("nonexistent", 0) == 0


def test_molecule_record_provenance(aspirin_record):
    aspirin_record.add_provenance("validated")
    aspirin_record.add_provenance("standardized")
    assert len(aspirin_record.provenance) == 2
    assert "validated" in aspirin_record.provenance


def test_molecule_record_to_dict(aspirin_record):
    aspirin_record.set_property("MolWt", 180.16)
    d = aspirin_record.to_dict()
    assert "smiles" in d
    assert "record_id" in d
    assert d["MolWt"] == 180.16


def test_dataset_creation(sample_dataset):
    assert len(sample_dataset) == 5
    assert sample_dataset.name == "test_dataset"


def test_dataset_valid_records(sample_dataset):
    valid = sample_dataset.valid_records
    failed = sample_dataset.failed_records
    assert len(valid) == 4  # aspirin, caffeine, ibuprofen, salt
    assert len(failed) == 1  # invalid


def test_dataset_filter(sample_dataset):
    filtered = sample_dataset.filter(
        lambda r: r.source_id != "salt",
        filter_name="remove_salt",
    )
    assert len(filtered) < len(sample_dataset)


def test_dataset_to_dataframe(sample_dataset):
    df = sample_dataset.to_dataframe()
    assert len(df) == 5
    assert "smiles" in df.columns
    assert "status" in df.columns


def test_dataset_from_dataframe():
    import pandas as pd
    df = pd.DataFrame({
        "smiles": ["CCO", "CC(=O)O"],
        "name": ["ethanol", "acetic_acid"],
    })
    ds = MoleculeDataset.from_dataframe(df, smiles_col="smiles")
    assert len(ds) == 2
    assert ds[0].is_valid


def test_dataset_summary(sample_dataset):
    s = sample_dataset.summary()
    assert s["total"] == 5
    assert s["valid"] == 4
    assert s["failed"] == 1

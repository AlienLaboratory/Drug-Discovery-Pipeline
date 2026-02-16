"""Tests for molecular validators."""

from rdkit import Chem

from claudedd.core.models import MoleculeRecord, MoleculeStatus
from claudedd.phase1.data.validators import validate_molecule, validate_dataset


def test_validate_valid_molecule(aspirin_record):
    validate_molecule(aspirin_record)
    assert aspirin_record.status == MoleculeStatus.VALIDATED
    assert aspirin_record.smiles  # canonical SMILES set


def test_validate_invalid_molecule():
    rec = MoleculeRecord(mol=None, smiles="INVALID")
    rec.add_error("parse failed")
    validate_molecule(rec)
    assert rec.status == MoleculeStatus.FAILED


def test_validate_sets_inchi(aspirin_record):
    validate_molecule(aspirin_record)
    assert aspirin_record.inchi.startswith("InChI=")
    assert aspirin_record.inchikey


def test_validate_dataset(sample_dataset):
    validate_dataset(sample_dataset)
    valid = [r for r in sample_dataset if r.status == MoleculeStatus.VALIDATED]
    failed = [r for r in sample_dataset if r.status == MoleculeStatus.FAILED]
    assert len(valid) >= 3
    assert len(failed) >= 1

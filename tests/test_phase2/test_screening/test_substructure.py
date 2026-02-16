"""Tests for substructure screening."""

import pytest
from rdkit import Chem

from claudedd.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus
from claudedd.core.exceptions import ScreeningError
from claudedd.phase2.screening.substructure import (
    parse_pattern,
    has_substructure,
    count_substructure_matches,
    get_substructure_matches,
    screen_substructure,
    screen_multi_substructure,
)


def test_parse_pattern_smarts():
    """SMARTS pattern is parsed correctly."""
    query = parse_pattern("[CX3](=O)[OX2H1]")  # carboxylic acid
    assert query is not None


def test_parse_pattern_smiles():
    """SMILES pattern is parsed correctly."""
    query = parse_pattern("c1ccccc1")  # benzene
    assert query is not None


def test_parse_pattern_invalid():
    """Invalid pattern raises ScreeningError."""
    with pytest.raises(ScreeningError):
        parse_pattern("NOT_VALID_XYZZY!!!")


def test_has_substructure_match(aspirin_mol):
    """Aspirin contains a carboxylic acid."""
    query = parse_pattern("C(=O)O")
    assert has_substructure(aspirin_mol, query) is True


def test_has_substructure_no_match(aspirin_mol):
    """Aspirin does not contain a chlorine."""
    query = parse_pattern("[Cl]")
    assert has_substructure(aspirin_mol, query) is False


def test_count_substructure_matches(aspirin_mol):
    """Aspirin has 2 carbonyl groups."""
    query = parse_pattern("C=O")
    count = count_substructure_matches(aspirin_mol, query)
    assert count >= 2


def test_get_substructure_matches(aspirin_mol):
    """Get atom indices of matches."""
    query = parse_pattern("C(=O)O")
    matches = get_substructure_matches(aspirin_mol, query)
    assert len(matches) > 0
    assert all(isinstance(m, tuple) for m in matches)


def test_screen_substructure(sample_dataset):
    """Screen dataset for carboxylic acid."""
    hits = screen_substructure(sample_dataset, pattern="C(=O)O")
    assert len(hits) > 0
    # Aspirin and ibuprofen have carboxylic acid
    hit_ids = {r.source_id for r in hits.valid_records}
    assert "aspirin" in hit_ids


def test_screen_substructure_exclude(sample_dataset):
    """Exclude mode selects molecules WITHOUT the pattern."""
    hits = screen_substructure(sample_dataset, pattern="C(=O)O", exclude=True)
    hit_ids = {r.source_id for r in hits.valid_records}
    assert "aspirin" not in hit_ids


def test_screen_substructure_count_matches(sample_dataset):
    """Count matches flag works."""
    hits = screen_substructure(
        sample_dataset, pattern="C=O", count_matches=True,
    )
    for rec in hits.valid_records:
        assert "substruct_match_count" in rec.properties


def test_screen_multi_substructure_any(sample_dataset):
    """Multi-pattern with ANY matching."""
    hits = screen_multi_substructure(
        sample_dataset,
        patterns=["C(=O)O", "n"],  # carboxylic acid OR nitrogen
        match_all=False,
    )
    # Aspirin (COOH), caffeine (N), ibuprofen (COOH)
    assert len(hits) >= 3


def test_screen_multi_substructure_all(sample_dataset):
    """Multi-pattern with ALL matching."""
    hits = screen_multi_substructure(
        sample_dataset,
        patterns=["C(=O)O", "c1ccccc1"],  # carboxylic acid AND benzene ring
        match_all=True,
    )
    # Aspirin and ibuprofen have both
    hit_ids = {r.source_id for r in hits.valid_records}
    assert "aspirin" in hit_ids

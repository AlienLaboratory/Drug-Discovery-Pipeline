"""Tests for pharmacophore screening."""

import pytest
from rdkit import Chem

from claudedd.core.exceptions import ScreeningError
from claudedd.phase2.screening.pharmacophore import (
    get_pharmacophore_features,
    get_available_feature_families,
    check_pharmacophore_requirements,
    screen_pharmacophore,
    parse_feature_string,
)


def test_get_available_families():
    """Feature families include standard types."""
    families = get_available_feature_families()
    assert "Donor" in families
    assert "Acceptor" in families
    assert "Aromatic" in families


def test_get_features_aspirin(aspirin_mol):
    """Aspirin has donors, acceptors, and aromatic."""
    feats = get_pharmacophore_features(aspirin_mol)
    assert feats.get("Acceptor", 0) > 0
    assert feats.get("Aromatic", 0) > 0


def test_get_features_none():
    """None molecule returns empty dict."""
    feats = get_pharmacophore_features(None)
    assert feats == {}


def test_check_requirements_pass(aspirin_mol):
    """Aspirin meets donor/acceptor requirements."""
    result = check_pharmacophore_requirements(
        aspirin_mol, {"Acceptor": 1, "Aromatic": 1}
    )
    assert result["Acceptor"] is True
    assert result["Aromatic"] is True


def test_check_requirements_fail(aspirin_mol):
    """Aspirin fails very high donor requirement."""
    result = check_pharmacophore_requirements(
        aspirin_mol, {"Donor": 100}
    )
    assert result["Donor"] is False


def test_screen_pharmacophore(sample_dataset):
    """Screen for molecules with acceptors."""
    hits = screen_pharmacophore(
        sample_dataset,
        required_features={"Acceptor": 1},
        match_all=True,
    )
    assert len(hits) > 0


def test_screen_pharmacophore_empty_raises():
    """Empty requirements raises ScreeningError."""
    from claudedd.core.models import MoleculeDataset
    ds = MoleculeDataset(records=[], name="empty")
    with pytest.raises(ScreeningError, match="must not be empty"):
        screen_pharmacophore(ds, required_features={})


def test_screen_pharmacophore_invalid_family():
    """Unknown feature family raises ScreeningError."""
    from claudedd.core.models import MoleculeDataset
    ds = MoleculeDataset(records=[], name="empty")
    with pytest.raises(ScreeningError, match="Unknown feature family"):
        screen_pharmacophore(ds, required_features={"FakeFamily": 1})


def test_parse_feature_string_with_counts():
    """Parse 'Donor:2,Acceptor:3'."""
    result = parse_feature_string("Donor:2,Acceptor:3")
    assert result == {"Donor": 2, "Acceptor": 3}


def test_parse_feature_string_no_counts():
    """Parse 'Donor,Acceptor' defaults to count=1."""
    result = parse_feature_string("Donor,Acceptor")
    assert result == {"Donor": 1, "Acceptor": 1}

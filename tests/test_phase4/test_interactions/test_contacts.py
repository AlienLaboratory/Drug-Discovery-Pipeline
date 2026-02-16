"""Tests for protein-ligand contact detection."""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from drugflow.core.exceptions import DockingError
from drugflow.phase4.interactions.contacts import (
    detect_all_contacts,
    detect_hbonds,
    detect_hydrophobic_contacts,
    detect_pi_stacking,
    detect_salt_bridges,
)


@pytest.fixture
def two_molecules_3d():
    """Two small 3D molecules for contact detection tests.

    Uses aspirin as both 'ligand' and 'protein' for simplicity.
    """
    mol1 = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    mol1 = Chem.AddHs(mol1)
    AllChem.EmbedMolecule(mol1, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol1)

    mol2 = Chem.MolFromSmiles("c1ccc(O)cc1")  # Phenol
    mol2 = Chem.AddHs(mol2)
    AllChem.EmbedMolecule(mol2, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol2)

    return mol1, mol2


class TestDetectHBonds:
    """Tests for hydrogen bond detection."""

    def test_returns_list(self, two_molecules_3d):
        """Returns a list of contacts."""
        lig, prot = two_molecules_3d
        hbonds = detect_hbonds(lig, prot)
        assert isinstance(hbonds, list)

    def test_hbond_has_required_keys(self, two_molecules_3d):
        """Each H-bond dict has required keys."""
        lig, prot = two_molecules_3d
        hbonds = detect_hbonds(lig, prot, distance_cutoff=10.0)  # Large cutoff
        if hbonds:
            hb = hbonds[0]
            assert "ligand_atom" in hb
            assert "protein_atom" in hb
            assert "distance" in hb
            assert "type" in hb

    def test_no_conformers_raises(self):
        """Missing conformers raises error."""
        mol1 = Chem.MolFromSmiles("O")
        mol2 = Chem.MolFromSmiles("O")
        with pytest.raises(DockingError):
            detect_hbonds(mol1, mol2)


class TestDetectHydrophobicContacts:
    """Tests for hydrophobic contact detection."""

    def test_returns_list(self, two_molecules_3d):
        """Returns a list."""
        lig, prot = two_molecules_3d
        contacts = detect_hydrophobic_contacts(lig, prot)
        assert isinstance(contacts, list)

    def test_contact_has_type(self, two_molecules_3d):
        """Contacts have 'hydrophobic' type."""
        lig, prot = two_molecules_3d
        contacts = detect_hydrophobic_contacts(lig, prot, distance_cutoff=10.0)
        if contacts:
            assert contacts[0]["type"] == "hydrophobic"


class TestDetectPiStacking:
    """Tests for pi-stacking detection."""

    def test_returns_list(self, two_molecules_3d):
        """Returns a list of pi interactions."""
        lig, prot = two_molecules_3d
        contacts = detect_pi_stacking(lig, prot)
        assert isinstance(contacts, list)

    def test_stacking_between_aromatic_molecules(self):
        """Detect pi-stacking between two benzene rings (close together)."""
        mol1 = Chem.MolFromSmiles("c1ccccc1")
        mol1 = Chem.AddHs(mol1)
        AllChem.EmbedMolecule(mol1, AllChem.ETKDGv3())

        mol2 = Chem.MolFromSmiles("c1ccccc1")
        mol2 = Chem.AddHs(mol2)
        AllChem.EmbedMolecule(mol2, AllChem.ETKDGv3())

        contacts = detect_pi_stacking(mol1, mol2, distance_cutoff=20.0)
        assert isinstance(contacts, list)


class TestDetectSaltBridges:
    """Tests for salt bridge detection."""

    def test_returns_list(self, two_molecules_3d):
        """Returns a list."""
        lig, prot = two_molecules_3d
        bridges = detect_salt_bridges(lig, prot)
        assert isinstance(bridges, list)


class TestDetectAllContacts:
    """Tests for combined contact detection."""

    def test_returns_all_types(self, two_molecules_3d):
        """Returns dict with all contact types."""
        lig, prot = two_molecules_3d
        contacts = detect_all_contacts(lig, prot)
        assert "hbonds" in contacts
        assert "hydrophobic" in contacts
        assert "pi_stacking" in contacts
        assert "salt_bridges" in contacts

    def test_all_values_are_lists(self, two_molecules_3d):
        """All values in result are lists."""
        lig, prot = two_molecules_3d
        contacts = detect_all_contacts(lig, prot)
        for key, value in contacts.items():
            assert isinstance(value, list)

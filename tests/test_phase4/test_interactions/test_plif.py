"""Tests for Protein-Ligand Interaction Fingerprints."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from claudedd.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus
from claudedd.phase4.interactions.plif import (
    compare_plif,
    compute_plif,
    compute_plif_dataset,
)


@pytest.fixture
def ligand_protein_pair():
    """A ligand-protein pair for PLIF tests."""
    lig = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    lig = Chem.AddHs(lig)
    AllChem.EmbedMolecule(lig, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(lig)

    prot = Chem.MolFromSmiles("c1ccc(O)cc1N")  # Amino-phenol as "protein"
    prot = Chem.AddHs(prot)
    AllChem.EmbedMolecule(prot, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(prot)

    return lig, prot


class TestComputePLIF:
    """Tests for PLIF computation."""

    def test_plif_returns_array(self, ligand_protein_pair):
        """PLIF returns numpy array."""
        lig, prot = ligand_protein_pair
        plif = compute_plif(lig, prot)
        assert isinstance(plif, np.ndarray)

    def test_plif_length(self, ligand_protein_pair):
        """PLIF has expected length (9 elements)."""
        lig, prot = ligand_protein_pair
        plif = compute_plif(lig, prot)
        assert len(plif) == 9

    def test_plif_nonnegative(self, ligand_protein_pair):
        """All PLIF values are non-negative."""
        lig, prot = ligand_protein_pair
        plif = compute_plif(lig, prot)
        assert np.all(plif >= 0)


class TestComputePLIFDataset:
    """Tests for dataset-level PLIF."""

    def test_dataset_plif(self, dataset_3d):
        """PLIF is computed for dataset molecules."""
        # Use first molecule as "protein"
        protein = dataset_3d.valid_records[0].mol
        result = compute_plif_dataset(dataset_3d, protein)
        has_plif = any("plif" in r.properties for r in result.valid_records)
        assert has_plif


class TestComparePLIF:
    """Tests for PLIF comparison."""

    def test_identical_plifs(self):
        """Identical PLIFs have similarity 1.0."""
        plif = np.array([1, 2, 3, 0, 1, 1, 1, 0, 1])
        assert compare_plif(plif, plif) == 1.0

    def test_empty_plifs(self):
        """Two empty PLIFs are identical."""
        plif = np.zeros(9)
        assert compare_plif(plif, plif) == 1.0

    def test_different_plifs(self):
        """Different PLIFs have similarity < 1."""
        plif1 = np.array([1, 0, 3, 0, 0, 1, 0, 0, 0])
        plif2 = np.array([0, 2, 0, 1, 0, 0, 1, 1, 0])
        sim = compare_plif(plif1, plif2)
        assert 0.0 <= sim < 1.0

    def test_similarity_range(self):
        """Similarity is in [0, 1] range."""
        plif1 = np.array([3, 1, 2, 0, 1, 1, 1, 0, 1])
        plif2 = np.array([0, 0, 5, 2, 0, 0, 1, 1, 0])
        sim = compare_plif(plif1, plif2)
        assert 0.0 <= sim <= 1.0

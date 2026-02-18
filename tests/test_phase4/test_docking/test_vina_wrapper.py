"""Tests for AutoDock Vina wrapper (Python bindings + CLI binary).

Covers availability detection, PDBQT conversion, output parsing,
and docking integration tests (which require Vina to be installed).
"""

import os
import tempfile

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from drugflow.core.exceptions import DockingError
from drugflow.phase4.docking.grid import DockingBox
from drugflow.phase4.docking.vina_wrapper import (
    _check_meeko_available,
    _check_vina_available,
    _check_vina_binary,
    _check_vina_python,
    _parse_vina_output,
    dock_vina,
    get_vina_backend,
    mol_to_pdbqt,
    protein_pdb_to_pdbqt,
)


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def aspirin_3d():
    """Aspirin molecule with 3D conformer and Hs."""
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


@pytest.fixture
def ethanol_3d():
    """Ethanol molecule with 3D conformer."""
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    return mol


@pytest.fixture
def sample_pdb(tmp_path):
    """Create a minimal PDB file for testing protein conversion."""
    pdb_content = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00  0.00           C
ATOM      4  O   ALA A   1       4.000   5.000   6.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       5.000   6.000   7.000  1.00  0.00           C
TER
END
"""
    pdb_path = str(tmp_path / "test_protein.pdb")
    with open(pdb_path, "w") as f:
        f.write(pdb_content)
    return pdb_path


# ── Availability checks ────────────────────────────────────────────

class TestVinaAvailability:
    """Tests for Vina/Meeko availability detection."""

    def test_check_vina_python_returns_bool(self):
        """Python bindings check returns bool."""
        result = _check_vina_python()
        assert isinstance(result, bool)

    def test_check_vina_binary_returns_str_or_none(self):
        """Binary check returns path string or None."""
        result = _check_vina_binary()
        assert result is None or isinstance(result, str)

    def test_check_vina_available_returns_bool(self):
        """Combined availability check returns bool."""
        result = _check_vina_available()
        assert isinstance(result, bool)

    def test_check_meeko_returns_bool(self):
        """Meeko check returns a boolean."""
        result = _check_meeko_available()
        assert isinstance(result, bool)

    def test_get_vina_backend(self):
        """Backend detection returns valid string."""
        result = get_vina_backend()
        assert result in ("python", "binary", "none")

    def test_vina_binary_is_found(self):
        """Vina binary should be found in tools/ directory."""
        vina_path = _check_vina_binary()
        if vina_path is None:
            pytest.skip("Vina binary not installed")
        assert os.path.isfile(vina_path)
        assert "vina" in os.path.basename(vina_path).lower()


# ── PDBQT conversion ───────────────────────────────────────────────

class TestMolToPdbqt:
    """Tests for ligand PDBQT conversion via Meeko."""

    def test_aspirin_conversion(self, aspirin_3d):
        """Aspirin converts to valid PDBQT string."""
        if not _check_meeko_available():
            pytest.skip("Meeko not installed")
        pdbqt = mol_to_pdbqt(aspirin_3d)
        assert isinstance(pdbqt, str)
        assert len(pdbqt) > 100
        assert "ATOM" in pdbqt or "HETATM" in pdbqt
        assert "ROOT" in pdbqt  # Meeko torsion tree

    def test_ethanol_conversion(self, ethanol_3d):
        """Small molecule also converts."""
        if not _check_meeko_available():
            pytest.skip("Meeko not installed")
        pdbqt = mol_to_pdbqt(ethanol_3d)
        assert isinstance(pdbqt, str)
        assert len(pdbqt) > 50

    def test_none_mol_raises(self):
        """None molecule raises DockingError."""
        with pytest.raises(DockingError):
            mol_to_pdbqt(None)

    def test_no_conformer_raises(self):
        """Molecule without conformer raises DockingError."""
        mol = Chem.MolFromSmiles("CCO")
        with pytest.raises(DockingError):
            mol_to_pdbqt(mol)


class TestProteinPdbToPdbqt:
    """Tests for protein PDB → PDBQT conversion."""

    def test_basic_conversion(self, sample_pdb, tmp_path):
        """PDB converts to PDBQT with atom types."""
        output = str(tmp_path / "test_protein.pdbqt")
        result = protein_pdb_to_pdbqt(sample_pdb, output)
        assert os.path.isfile(result)
        with open(result) as f:
            content = f.read()
        # Should contain ATOM lines with AD types appended
        assert "ATOM" in content
        # Should contain AutoDock atom types (N, C, OA)
        lines = [l for l in content.split("\n") if l.startswith("ATOM")]
        assert len(lines) == 5  # 5 atoms

    def test_default_output_path(self, sample_pdb):
        """No output_path → writes .pdbqt next to .pdb."""
        result = protein_pdb_to_pdbqt(sample_pdb)
        expected = sample_pdb.replace(".pdb", ".pdbqt")
        assert result == expected
        assert os.path.isfile(result)
        # Cleanup
        os.unlink(result)

    def test_nonexistent_pdb_raises(self):
        """Missing PDB file raises DockingError."""
        with pytest.raises(DockingError):
            protein_pdb_to_pdbqt("/nonexistent/protein.pdb")


# ── Vina output parsing ────────────────────────────────────────────

class TestParseVinaOutput:
    """Tests for parsing Vina CLI stdout."""

    def test_typical_output(self):
        """Parse typical Vina output with multiple poses."""
        stdout = """\
Computing Vina grid ... done.
Performing docking (random seed: 1234) ...
mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1       -8.3          0.000      0.000
   2       -7.9          1.234      2.345
   3       -7.1          3.456      5.678
"""
        results = _parse_vina_output(stdout)
        assert len(results) == 3
        assert results[0]["pose_id"] == 0
        assert results[0]["vina_score"] == -8.3
        assert results[0]["rmsd_lb"] == 0.0
        assert results[1]["vina_score"] == -7.9
        assert results[2]["vina_score"] == -7.1

    def test_single_pose(self):
        """Parse output with just one pose."""
        stdout = """\
-----+------------+----------+----------
   1       -6.5          0.000      0.000
"""
        results = _parse_vina_output(stdout)
        assert len(results) == 1
        assert results[0]["vina_score"] == -6.5

    def test_empty_output(self):
        """Empty output returns empty list."""
        results = _parse_vina_output("")
        assert results == []

    def test_no_table(self):
        """Output without table header returns empty list."""
        stdout = "Some random Vina output\nwithout results\n"
        results = _parse_vina_output(stdout)
        assert results == []


# ── Integration docking tests ──────────────────────────────────────

class TestDockVina:
    """Integration tests (require Vina to be installed)."""

    def test_no_vina_raises_docking_error(self):
        """When Vina not available, dock_vina raises DockingError."""
        if _check_vina_available():
            pytest.skip("Vina is available — cannot test missing-vina path")

        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

        box = DockingBox(0, 0, 0, 20, 20, 20)
        with pytest.raises(DockingError):
            dock_vina(mol, "dummy_protein.pdbqt", box)

    def test_none_ligand_raises(self):
        """None ligand raises error."""
        if not _check_vina_available():
            pytest.skip("Vina not installed")
        box = DockingBox(0, 0, 0, 20, 20, 20)
        with pytest.raises(DockingError):
            dock_vina(None, "protein.pdbqt", box)

    def test_missing_protein_raises(self):
        """Missing protein file raises error."""
        if not _check_vina_available():
            pytest.skip("Vina not installed")

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

        box = DockingBox(0, 0, 0, 20, 20, 20)
        with pytest.raises(DockingError):
            dock_vina(mol, "/nonexistent/protein.pdbqt", box)

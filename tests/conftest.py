"""Shared test fixtures."""

import os

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from drugflow.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE_SMILES = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
IBUPROFEN_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
INVALID_SMILES = "XYZNOTAMOLECULE"
SALT_SMILES = "CC(=O)[O-].[Na+]"
PAINS_SMILES = "O=C1C=CC(=O)C=C1"  # Benzoquinone

SAMPLE_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "sample", "sample_molecules.csv"
)
SAMPLE_SMI = os.path.join(
    os.path.dirname(__file__), "..", "data", "sample", "sample_molecules.smi"
)


@pytest.fixture
def aspirin_mol():
    return Chem.MolFromSmiles(ASPIRIN_SMILES)


@pytest.fixture
def aspirin_record():
    mol = Chem.MolFromSmiles(ASPIRIN_SMILES)
    return MoleculeRecord(
        mol=mol, smiles=ASPIRIN_SMILES,
        source_id="aspirin", status=MoleculeStatus.RAW,
    )


@pytest.fixture
def sample_dataset():
    """A dataset of 5 molecules including one invalid and one salt."""
    smiles_list = [
        ("aspirin", ASPIRIN_SMILES),
        ("caffeine", CAFFEINE_SMILES),
        ("ibuprofen", IBUPROFEN_SMILES),
        ("invalid", INVALID_SMILES),
        ("salt", SALT_SMILES),
    ]
    records = []
    for name, smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        rec = MoleculeRecord(
            mol=mol, smiles=smi,
            source_id=name, status=MoleculeStatus.RAW,
        )
        if mol is None:
            rec.add_error(f"Failed to parse: {smi}")
        records.append(rec)
    return MoleculeDataset(records=records, name="test_dataset")


@pytest.fixture
def sample_csv_path(tmp_path):
    """Create a temporary CSV file."""
    path = tmp_path / "test.csv"
    path.write_text(
        "id,smiles,activity\n"
        "aspirin,CC(=O)Oc1ccccc1C(=O)O,5.2\n"
        "caffeine,Cn1c(=O)c2c(ncn2C)n(C)c1=O,4.1\n"
        "ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,6.3\n"
    )
    return str(path)


@pytest.fixture
def sample_smi_path(tmp_path):
    """Create a temporary SMILES file."""
    path = tmp_path / "test.smi"
    path.write_text(
        "CC(=O)Oc1ccccc1C(=O)O\taspirin\n"
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O\tcaffeine\n"
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O\tibuprofen\n"
    )
    return str(path)


@pytest.fixture
def sample_sdf_path(tmp_path):
    """Create a temporary SDF file."""
    path = tmp_path / "test.sdf"
    writer = Chem.SDWriter(str(path))
    for smi, name in [
        (ASPIRIN_SMILES, "aspirin"),
        (CAFFEINE_SMILES, "caffeine"),
        (IBUPROFEN_SMILES, "ibuprofen"),
    ]:
        mol = Chem.MolFromSmiles(smi)
        AllChem.Compute2DCoords(mol)
        mol.SetProp("_Name", name)
        writer.write(mol)
    writer.close()
    return str(path)


# ── Phase 2 Fixtures ──────────────────────────────────────────

# Additional drug molecules for richer test datasets
METFORMIN_SMILES = "CN(C)C(=N)NC(=N)N"
ATORVASTATIN_SMILES = "CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(=O)[O-])c(-c2ccccc2)c(-c2ccc(F)cc2)c1C(=O)Nc1ccccc1"
NAPROXEN_SMILES = "COc1ccc2cc(C(C)C(=O)O)ccc2c1"
DICLOFENAC_SMILES = "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl"
CELECOXIB_SMILES = "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"


@pytest.fixture
def activity_dataset():
    """Dataset with activity values for QSAR testing (10 molecules)."""
    molecules = [
        ("aspirin", ASPIRIN_SMILES, 5.2),
        ("caffeine", CAFFEINE_SMILES, 4.1),
        ("ibuprofen", IBUPROFEN_SMILES, 6.3),
        ("naproxen", NAPROXEN_SMILES, 6.8),
        ("diclofenac", DICLOFENAC_SMILES, 7.1),
        ("metformin", METFORMIN_SMILES, 3.5),
        ("celecoxib", CELECOXIB_SMILES, 7.5),
        ("salt", SALT_SMILES, 2.0),
        ("pains_mol", PAINS_SMILES, 1.5),
        ("aspirin2", ASPIRIN_SMILES, 5.0),  # duplicate for scaffold split test
    ]
    records = []
    for name, smi, activity in molecules:
        mol = Chem.MolFromSmiles(smi)
        rec = MoleculeRecord(
            mol=mol, smiles=smi,
            source_id=name, status=MoleculeStatus.RAW,
        )
        if mol is None:
            rec.add_error(f"Failed to parse: {smi}")
        else:
            rec.metadata["activity"] = activity
        records.append(rec)
    return MoleculeDataset(records=records, name="activity_test_dataset")


@pytest.fixture
def computed_dataset():
    """Dataset with properties, descriptors, and fingerprints pre-computed."""
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.analysis.properties import compute_properties_dataset
    from drugflow.phase1.analysis.descriptors import compute_descriptors_dataset
    from drugflow.phase1.analysis.fingerprints import compute_fingerprints_dataset

    molecules = [
        ("aspirin", ASPIRIN_SMILES, 5.2),
        ("caffeine", CAFFEINE_SMILES, 4.1),
        ("ibuprofen", IBUPROFEN_SMILES, 6.3),
        ("naproxen", NAPROXEN_SMILES, 6.8),
        ("diclofenac", DICLOFENAC_SMILES, 7.1),
    ]
    records = []
    for name, smi, activity in molecules:
        mol = Chem.MolFromSmiles(smi)
        rec = MoleculeRecord(
            mol=mol, smiles=smi,
            source_id=name, status=MoleculeStatus.RAW,
        )
        rec.metadata["activity"] = activity
        records.append(rec)

    dataset = MoleculeDataset(records=records, name="computed_dataset")
    dataset = validate_dataset(dataset)
    dataset = compute_properties_dataset(dataset)
    dataset = compute_descriptors_dataset(dataset)
    dataset = compute_fingerprints_dataset(dataset)
    return dataset


@pytest.fixture
def activity_csv_path(tmp_path):
    """CSV file with activity data for QSAR testing."""
    path = tmp_path / "activity_data.csv"
    path.write_text(
        "id,smiles,activity\n"
        "aspirin,CC(=O)Oc1ccccc1C(=O)O,5.2\n"
        "caffeine,Cn1c(=O)c2c(ncn2C)n(C)c1=O,4.1\n"
        "ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,6.3\n"
        "naproxen,COc1ccc2cc(C(C)C(=O)O)ccc2c1,6.8\n"
        "diclofenac,OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl,7.1\n"
        "metformin,CN(C)C(=N)NC(=N)N,3.5\n"
        "celecoxib,Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1,7.5\n"
        "pains_mol,O=C1C=CC(=O)C=C1,1.5\n"
    )
    return str(path)


# ── Phase 3 Fixtures ──────────────────────────────────────────

@pytest.fixture
def seed_dataset():
    """Small dataset suitable for generation (all valid molecules with rings)."""
    molecules = [
        ("aspirin", ASPIRIN_SMILES),
        ("caffeine", CAFFEINE_SMILES),
        ("ibuprofen", IBUPROFEN_SMILES),
        ("naproxen", NAPROXEN_SMILES),
        ("diclofenac", DICLOFENAC_SMILES),
    ]
    records = []
    for name, smi in molecules:
        mol = Chem.MolFromSmiles(smi)
        rec = MoleculeRecord(
            mol=mol, smiles=smi,
            source_id=name, status=MoleculeStatus.RAW,
        )
        records.append(rec)
    return MoleculeDataset(records=records, name="seed_dataset")


@pytest.fixture
def trained_rf_model(computed_dataset):
    """Trained Random Forest model for GA and active learning tests."""
    from drugflow.phase2.qsar.data_prep import extract_feature_matrix, extract_labels
    from drugflow.phase2.qsar.models import train_model

    X, names, indices = extract_feature_matrix(
        computed_dataset, feature_source="descriptors",
    )
    y = extract_labels(computed_dataset, "activity", indices)
    return train_model(X, y, feature_names=names, task="regression")


# ── Phase 4 Fixtures ──────────────────────────────────────────

@pytest.fixture
def aspirin_3d():
    """Aspirin molecule with 3D coordinates and hydrogens."""
    mol = Chem.MolFromSmiles(ASPIRIN_SMILES)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


@pytest.fixture
def ibuprofen_3d():
    """Ibuprofen molecule with 3D coordinates and hydrogens."""
    mol = Chem.MolFromSmiles(IBUPROFEN_SMILES)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


@pytest.fixture
def caffeine_3d():
    """Caffeine molecule with 3D coordinates and hydrogens."""
    mol = Chem.MolFromSmiles(CAFFEINE_SMILES)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


@pytest.fixture
def multi_conf_mol():
    """Aspirin with multiple conformers for conformer-related tests."""
    mol = Chem.MolFromSmiles(ASPIRIN_SMILES)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMultipleConfs(mol, numConfs=5, params=params)
    return mol


@pytest.fixture
def dataset_3d():
    """Dataset with 3D-prepared molecules."""
    smiles_list = [
        ("aspirin", ASPIRIN_SMILES),
        ("caffeine", CAFFEINE_SMILES),
        ("ibuprofen", IBUPROFEN_SMILES),
    ]
    records = []
    for name, smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        rec = MoleculeRecord(
            mol=mol, smiles=smi,
            source_id=name, status=MoleculeStatus.RAW,
        )
        records.append(rec)
    return MoleculeDataset(records=records, name="dataset_3d")

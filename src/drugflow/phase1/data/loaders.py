"""File loaders for molecular data formats.

Each loader reads a file and returns a MoleculeDataset containing
MoleculeRecords in RAW status.

Supported formats: SDF, SMILES (.smi), CSV, TSV, PDB
"""

import gzip
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem

from drugflow.core.exceptions import FileFormatError
from drugflow.core.logging import get_logger, progress_bar
from drugflow.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus
from drugflow.utils.io import detect_format

logger = get_logger("data.loaders")


def load_file(
    path: str,
    format: str = "auto",
    smiles_column: str = "smiles",
    id_column: Optional[str] = None,
    limit: Optional[int] = None,
) -> MoleculeDataset:
    """Universal file loader. Auto-detects format from extension."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if format == "auto":
        format = detect_format(path)

    loaders = {
        "sdf": lambda: load_sdf(path, limit=limit),
        "smi": lambda: load_smiles(path, limit=limit),
        "csv": lambda: load_csv(path, smiles_column=smiles_column,
                                id_column=id_column, delimiter=",", limit=limit),
        "tsv": lambda: load_csv(path, smiles_column=smiles_column,
                                id_column=id_column, delimiter="\t", limit=limit),
        "pdb": lambda: load_pdb(path),
    }

    loader = loaders.get(format)
    if loader is None:
        raise FileFormatError(f"No loader for format: {format}")

    dataset = loader()
    dataset.name = p.stem
    logger.info(f"Loaded {len(dataset)} molecules from {path} (format={format})")
    return dataset


def load_sdf(
    path: str,
    limit: Optional[int] = None,
) -> MoleculeDataset:
    """Load molecules from an SDF file."""
    records = []

    is_gzipped = path.endswith(".gz")
    if is_gzipped:
        supplier = Chem.ForwardSDMolSupplier(gzip.open(path))
    else:
        supplier = Chem.SDMolSupplier(path)

    for i, mol in enumerate(progress_bar(supplier, desc="Loading SDF")):
        if limit is not None and i >= limit:
            break

        rec = MoleculeRecord(
            mol=mol,
            source_file=path,
            source_index=i,
            status=MoleculeStatus.RAW,
        )

        if mol is None:
            rec.add_error(f"Failed to parse molecule at index {i}")
        else:
            # Preserve SDF properties as metadata
            for prop_name in mol.GetPropsAsDict():
                rec.metadata[prop_name] = mol.GetPropsAsDict()[prop_name]
            # Try to get name
            if mol.HasProp("_Name"):
                rec.source_id = mol.GetProp("_Name")

        rec.add_provenance("loaded:sdf")
        records.append(rec)

    return MoleculeDataset(records=records)


def load_smiles(
    path: str,
    delimiter: str = "\t",
    has_header: bool = False,
    limit: Optional[int] = None,
) -> MoleculeDataset:
    """Load molecules from a SMILES file (.smi)."""
    records = []

    with open(path, "r") as f:
        lines = f.readlines()

    start = 1 if has_header else 0

    for i, line in enumerate(progress_bar(lines[start:], desc="Loading SMILES")):
        if limit is not None and i >= limit:
            break

        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(delimiter)
        smi = parts[0].strip()
        name = parts[1].strip() if len(parts) > 1 else ""

        mol = Chem.MolFromSmiles(smi)
        rec = MoleculeRecord(
            mol=mol,
            smiles=smi,
            source_id=name,
            source_file=path,
            source_index=i,
            status=MoleculeStatus.RAW,
        )

        if mol is None:
            rec.add_error(f"Failed to parse SMILES: {smi}")

        rec.add_provenance("loaded:smi")
        records.append(rec)

    return MoleculeDataset(records=records)


def load_csv(
    path: str,
    smiles_column: str = "smiles",
    id_column: Optional[str] = None,
    delimiter: str = ",",
    limit: Optional[int] = None,
) -> MoleculeDataset:
    """Load molecules from a CSV/TSV file."""
    df = pd.read_csv(path, delimiter=delimiter, nrows=limit)

    if smiles_column not in df.columns:
        # Try case-insensitive match
        col_map = {c.lower(): c for c in df.columns}
        if smiles_column.lower() in col_map:
            smiles_column = col_map[smiles_column.lower()]
        else:
            # Auto-detect common SMILES column names
            common_names = [
                "canonical_smiles", "smiles", "smi", "SMILES",
                "Canonical_SMILES", "molecule_smiles", "mol_smiles",
                "structure", "isosmiles", "isomeric_smiles",
            ]
            found = None
            for name in common_names:
                if name in df.columns:
                    found = name
                    break
                if name.lower() in col_map:
                    found = col_map[name.lower()]
                    break
            if found:
                smiles_column = found
                logger.info(
                    f"Auto-detected SMILES column: '{smiles_column}'"
                )
            else:
                raise FileFormatError(
                    f"SMILES column '{smiles_column}' not found in {path}. "
                    f"Available columns: {list(df.columns)}"
                )

    records = []
    for idx, row in progress_bar(df.iterrows(), total=len(df), desc="Loading CSV"):
        smi = str(row[smiles_column]) if pd.notna(row[smiles_column]) else ""
        mol = Chem.MolFromSmiles(smi) if smi else None

        source_id = ""
        if id_column and id_column in df.columns:
            source_id = str(row[id_column])

        rec = MoleculeRecord(
            mol=mol,
            smiles=smi,
            source_id=source_id,
            source_file=path,
            source_index=int(idx),
            status=MoleculeStatus.RAW,
        )

        if mol is None:
            rec.add_error(f"Failed to parse SMILES: {smi}")

        # Store extra columns as metadata
        for col in df.columns:
            if col not in (smiles_column, id_column):
                val = row[col]
                rec.metadata[col] = val if pd.notna(val) else None

        rec.add_provenance("loaded:csv")
        records.append(rec)

    return MoleculeDataset(records=records)


def load_pdb(path: str) -> MoleculeDataset:
    """Load a single molecule from a PDB file."""
    mol = Chem.MolFromPDBFile(path, removeHs=True, sanitize=True)

    rec = MoleculeRecord(
        mol=mol,
        source_file=path,
        source_index=0,
        status=MoleculeStatus.RAW,
    )

    if mol is None:
        rec.add_error(f"Failed to parse PDB file: {path}")
    else:
        rec.source_id = Path(path).stem

    rec.add_provenance("loaded:pdb")
    return MoleculeDataset(records=[rec], name=Path(path).stem)

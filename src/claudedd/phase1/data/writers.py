"""File writers for exporting MoleculeDataset to various formats."""

import json
from pathlib import Path
from typing import Optional

from rdkit import Chem

from claudedd.core.logging import get_logger, progress_bar
from claudedd.core.models import MoleculeDataset
from claudedd.utils.io import detect_format, ensure_output_dir

logger = get_logger("data.writers")


def write_file(
    dataset: MoleculeDataset,
    path: str,
    format: str = "auto",
    include_properties: bool = True,
) -> str:
    """Universal writer. Returns the output path."""
    if format == "auto":
        format = detect_format(path)

    ensure_output_dir(path)

    writers = {
        "sdf": lambda: write_sdf(dataset, path, include_properties),
        "csv": lambda: write_csv(dataset, path, include_properties),
        "tsv": lambda: write_csv(dataset, path, include_properties, delimiter="\t"),
        "smi": lambda: write_smiles(dataset, path),
        "json": lambda: write_json(dataset, path),
    }

    writer = writers.get(format)
    if writer is None:
        raise ValueError(f"Unsupported output format: {format}")

    result = writer()
    logger.info(f"Wrote {len(dataset)} molecules to {result}")
    return result


def write_sdf(
    dataset: MoleculeDataset,
    path: str,
    include_properties: bool = True,
) -> str:
    """Write molecules to SDF format."""
    ensure_output_dir(path)
    writer = Chem.SDWriter(path)

    for rec in progress_bar(dataset.records, desc="Writing SDF"):
        if rec.mol is None:
            continue

        mol = Chem.RWMol(rec.mol)

        if include_properties:
            for key, val in rec.properties.items():
                if val is not None:
                    mol.SetProp(str(key), str(val))
            if rec.source_id:
                mol.SetProp("_Name", rec.source_id)

        writer.write(mol)

    writer.close()
    return path


def write_csv(
    dataset: MoleculeDataset,
    path: str,
    include_properties: bool = True,
    delimiter: str = ",",
) -> str:
    """Write dataset to CSV."""
    ensure_output_dir(path)
    df = dataset.to_dataframe()
    df.to_csv(path, sep=delimiter, index=False)
    return path


def write_smiles(
    dataset: MoleculeDataset,
    path: str,
) -> str:
    """Write SMILES strings, one per line."""
    ensure_output_dir(path)

    with open(path, "w") as f:
        for rec in dataset.records:
            if rec.is_valid and rec.canonical_smiles:
                name = rec.source_id or rec.record_id
                f.write(f"{rec.canonical_smiles}\t{name}\n")

    return path


def write_json(
    dataset: MoleculeDataset,
    path: str,
    indent: int = 2,
) -> str:
    """Write dataset as JSON with full provenance."""
    ensure_output_dir(path)

    data = {
        "name": dataset.name,
        "summary": dataset.summary(),
        "records": [rec.to_dict() for rec in dataset.records],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)

    return path

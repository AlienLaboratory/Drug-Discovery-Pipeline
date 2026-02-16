"""Path resolution and file format detection utilities."""

from pathlib import Path
from typing import Optional

from drugflow.core.exceptions import FileFormatError

EXTENSION_MAP = {
    ".sdf": "sdf",
    ".sd": "sdf",
    ".smi": "smi",
    ".smiles": "smi",
    ".csv": "csv",
    ".tsv": "tsv",
    ".pdb": "pdb",
    ".mol": "mol",
    ".mol2": "mol2",
}


def detect_format(path: str) -> str:
    p = Path(path)

    # Handle .sdf.gz
    if p.suffixes == [".sdf", ".gz"]:
        return "sdf"

    ext = p.suffix.lower()
    fmt = EXTENSION_MAP.get(ext)
    if fmt is None:
        raise FileFormatError(
            f"Unsupported file extension '{ext}' for file: {path}. "
            f"Supported: {', '.join(sorted(EXTENSION_MAP.keys()))}"
        )
    return fmt


def ensure_output_dir(path: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(path: str) -> Path:
    return Path(path).resolve()

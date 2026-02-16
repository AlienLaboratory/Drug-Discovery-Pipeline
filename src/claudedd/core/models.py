"""Core data models for the ClaudeDD pipeline."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi as rdInchi


class MoleculeStatus(Enum):
    """Processing state of a molecule."""
    RAW = "raw"
    VALIDATED = "validated"
    STANDARDIZED = "standardized"
    FAILED = "failed"
    FILTERED = "filtered"


@dataclass
class MoleculeRecord:
    """Central data object wrapping an RDKit Mol with metadata and provenance."""

    mol: Optional[Chem.Mol]
    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_id: str = ""
    source_file: str = ""
    source_index: int = -1

    smiles: str = ""
    inchi: str = ""
    inchikey: str = ""

    properties: Dict[str, Any] = field(default_factory=dict)
    fingerprints: Dict[str, Any] = field(default_factory=dict)
    descriptors: Dict[str, float] = field(default_factory=dict)

    status: MoleculeStatus = MoleculeStatus.RAW
    errors: List[str] = field(default_factory=list)
    provenance: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.mol is not None and self.status != MoleculeStatus.FAILED

    @property
    def canonical_smiles(self) -> str:
        if not self.smiles and self.mol is not None:
            self.smiles = Chem.MolToSmiles(self.mol, canonical=True)
        return self.smiles

    def get_property(self, name: str, default: Any = None) -> Any:
        return self.properties.get(name, default)

    def set_property(self, name: str, value: Any) -> None:
        self.properties[name] = value

    def add_provenance(self, step: str) -> None:
        self.provenance.append(step)

    def add_error(self, error: str) -> None:
        self.errors.append(error)
        self.status = MoleculeStatus.FAILED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_id": self.source_id,
            "smiles": self.canonical_smiles,
            "inchi": self.inchi,
            "inchikey": self.inchikey,
            "status": self.status.value,
            "errors": "; ".join(self.errors),
            **self.properties,
            **self.metadata,
        }

    def __repr__(self) -> str:
        smi = self.smiles[:40] if self.smiles else "N/A"
        return f"MoleculeRecord(id={self.record_id}, smiles='{smi}', status={self.status.value})"


class MoleculeDataset:
    """Ordered collection of MoleculeRecords with batch operations."""

    def __init__(
        self,
        records: Optional[List[MoleculeRecord]] = None,
        name: str = "",
        description: str = "",
    ):
        self._records: List[MoleculeRecord] = records or []
        self.name: str = name
        self.description: str = description
        self._provenance: List[str] = []

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[MoleculeRecord]:
        return iter(self._records)

    def __getitem__(self, index: int) -> MoleculeRecord:
        return self._records[index]

    @property
    def records(self) -> List[MoleculeRecord]:
        return self._records

    @property
    def valid_records(self) -> List[MoleculeRecord]:
        return [r for r in self._records if r.is_valid]

    @property
    def failed_records(self) -> List[MoleculeRecord]:
        return [r for r in self._records if not r.is_valid]

    def add(self, record: MoleculeRecord) -> None:
        self._records.append(record)

    def extend(self, records: List[MoleculeRecord]) -> None:
        self._records.extend(records)

    def filter(
        self,
        predicate: Callable[[MoleculeRecord], bool],
        filter_name: str = "custom",
    ) -> "MoleculeDataset":
        passing = []
        for rec in self._records:
            if predicate(rec):
                passing.append(rec)
            else:
                rec.status = MoleculeStatus.FILTERED
                rec.add_provenance(f"filter:{filter_name}:fail")
        result = MoleculeDataset(records=passing, name=self.name)
        result._provenance = self._provenance + [f"filter:{filter_name}"]
        return result

    def to_dataframe(self, include_fingerprints: bool = False) -> pd.DataFrame:
        rows = [rec.to_dict() for rec in self._records]
        return pd.DataFrame(rows)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        smiles_col: str = "smiles",
        id_col: Optional[str] = None,
    ) -> "MoleculeDataset":
        records = []
        for idx, row in df.iterrows():
            smi = row.get(smiles_col, "")
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
            rec = MoleculeRecord(
                mol=mol,
                smiles=smi if isinstance(smi, str) else "",
                source_id=str(row[id_col]) if id_col and id_col in row.index else "",
                source_index=int(idx),
            )
            if mol is None:
                rec.add_error(f"Failed to parse SMILES: {smi}")
            for col in df.columns:
                if col not in (smiles_col, id_col):
                    rec.metadata[col] = row[col]
            records.append(rec)
        return cls(records=records)

    def summary(self) -> Dict[str, Any]:
        return {
            "total": len(self._records),
            "valid": len(self.valid_records),
            "failed": len(self.failed_records),
            "properties_computed": bool(
                self._records and self._records[0].properties
            ),
            "provenance": self._provenance,
        }

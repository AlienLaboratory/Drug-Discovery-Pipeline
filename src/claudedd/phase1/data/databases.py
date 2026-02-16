"""Utilities for fetching molecules from public databases.

Databases supported:
- ChEMBL (via REST API)
- PubChem (via PUG REST)
"""

import json
import urllib.request
import urllib.error
from typing import Optional

from rdkit import Chem

from claudedd.core.exceptions import DatabaseError
from claudedd.core.logging import get_logger
from claudedd.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus

logger = get_logger("data.databases")

CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"
PUBCHEM_API_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def fetch_chembl_by_target(
    target_id: str,
    activity_type: str = "IC50",
    max_results: int = 1000,
) -> MoleculeDataset:
    """Fetch bioactivity data from ChEMBL for a given target."""
    url = (
        f"{CHEMBL_API_BASE}/activity.json?"
        f"target_chembl_id={target_id}"
        f"&standard_type={activity_type}"
        f"&limit={min(max_results, 1000)}"
        f"&offset=0"
    )

    records = []
    try:
        data = _fetch_json(url)
        activities = data.get("activities", [])

        for i, act in enumerate(activities[:max_results]):
            smi = act.get("canonical_smiles", "")
            mol = Chem.MolFromSmiles(smi) if smi else None

            rec = MoleculeRecord(
                mol=mol,
                smiles=smi,
                source_id=act.get("molecule_chembl_id", ""),
                source_file=f"chembl:{target_id}",
                source_index=i,
                status=MoleculeStatus.RAW,
            )

            if mol is None:
                rec.add_error(f"Failed to parse SMILES: {smi}")

            # Store activity data as metadata
            rec.metadata["activity_type"] = act.get("standard_type", "")
            rec.metadata["activity_value"] = act.get("standard_value", "")
            rec.metadata["activity_units"] = act.get("standard_units", "")
            rec.metadata["activity_relation"] = act.get("standard_relation", "")
            rec.metadata["assay_chembl_id"] = act.get("assay_chembl_id", "")
            rec.metadata["target_chembl_id"] = target_id

            rec.add_provenance(f"fetched:chembl:{target_id}")
            records.append(rec)

        logger.info(
            f"Fetched {len(records)} compounds from ChEMBL "
            f"for target {target_id} ({activity_type})"
        )

    except Exception as e:
        raise DatabaseError(f"ChEMBL fetch failed for {target_id}: {e}")

    return MoleculeDataset(records=records, name=f"chembl_{target_id}")


def fetch_pubchem_by_name(
    name: str,
    max_results: int = 100,
) -> MoleculeDataset:
    """Search PubChem by compound name and fetch structures."""
    url = (
        f"{PUBCHEM_API_BASE}/compound/name/{urllib.request.quote(name)}"
        f"/property/CanonicalSMILES,MolecularFormula,MolecularWeight,"
        f"IUPACName,IsomericSMILES/JSON"
    )

    records = []
    try:
        data = _fetch_json(url)
        properties = data.get("PropertyTable", {}).get("Properties", [])

        for i, prop in enumerate(properties[:max_results]):
            smi = prop.get("CanonicalSMILES", "")
            mol = Chem.MolFromSmiles(smi) if smi else None

            rec = MoleculeRecord(
                mol=mol,
                smiles=smi,
                source_id=str(prop.get("CID", "")),
                source_file=f"pubchem:{name}",
                source_index=i,
                status=MoleculeStatus.RAW,
            )

            if mol is None:
                rec.add_error(f"Failed to parse SMILES: {smi}")

            rec.metadata["molecular_formula"] = prop.get("MolecularFormula", "")
            rec.metadata["molecular_weight"] = prop.get("MolecularWeight", "")
            rec.metadata["iupac_name"] = prop.get("IUPACName", "")

            rec.add_provenance(f"fetched:pubchem:{name}")
            records.append(rec)

        logger.info(f"Fetched {len(records)} compounds from PubChem for '{name}'")

    except Exception as e:
        raise DatabaseError(f"PubChem fetch failed for '{name}': {e}")

    return MoleculeDataset(records=records, name=f"pubchem_{name}")


def _fetch_json(url: str, timeout: int = 30) -> dict:
    """Fetch JSON from a URL."""
    req = urllib.request.Request(
        url, headers={"Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise DatabaseError(f"HTTP {e.code} error fetching {url}: {e.reason}")
    except urllib.error.URLError as e:
        raise DatabaseError(f"URL error fetching {url}: {e.reason}")

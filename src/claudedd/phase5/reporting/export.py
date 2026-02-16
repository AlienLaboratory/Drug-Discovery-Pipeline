"""Project export utilities for pipeline results.

Exports pipeline results to JSON and CSV formats with
provenance tracking for reproducibility.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from claudedd.core.models import MoleculeDataset

logger = logging.getLogger(__name__)


def export_project_json(
    report: Dict[str, Any],
    output_path: str,
) -> str:
    """Export a pipeline report as JSON.

    Args:
        report: Pipeline report dict (from create_pipeline_report).
        output_path: Path to output JSON file.

    Returns:
        Path to written file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report exported to {output_path}")
    return output_path


def export_results_csv(
    dataset: MoleculeDataset,
    output_path: str,
    columns: Optional[List[str]] = None,
) -> str:
    """Export dataset results to CSV with selected columns.

    Args:
        dataset: Dataset to export.
        output_path: Path to output CSV.
        columns: Specific property columns to include.
            If None, exports all available properties.

    Returns:
        Path to written file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Default columns for drug discovery export
    if columns is None:
        columns = [
            "MolWt", "LogP", "TPSA", "HBD", "HBA", "QED",
            "sa_score", "drug_likeness_score", "composite_score",
            "lipinski_pass", "pains_pass",
        ]

    rows = []
    for rec in dataset.valid_records:
        row = {
            "id": rec.source_id or rec.record_id,
            "smiles": rec.canonical_smiles or rec.smiles or "",
        }
        for col in columns:
            if col in rec.properties:
                val = rec.properties[col]
                if isinstance(val, float):
                    val = round(val, 4)
                row[col] = val
        rows.append(row)

    if not rows:
        logger.warning("No records to export")
        # Write empty CSV with header
        with open(output_path, "w") as f:
            f.write("id,smiles\n")
        return output_path

    # Write CSV
    all_keys = list(rows[0].keys())
    for row in rows[1:]:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)

    with open(output_path, "w") as f:
        f.write(",".join(all_keys) + "\n")
        for row in rows:
            vals = [str(row.get(k, "")) for k in all_keys]
            f.write(",".join(vals) + "\n")

    logger.info(f"Exported {len(rows)} molecules to {output_path}")
    return output_path


def create_provenance_record(dataset: MoleculeDataset) -> Dict[str, Any]:
    """Create a provenance record tracking which phases/tools were applied.

    Args:
        dataset: Dataset with provenance metadata.

    Returns:
        Dict with provenance info for each molecule.
    """
    provenance = {
        "timestamp": datetime.now().isoformat(),
        "dataset_name": dataset.name,
        "n_records": len(dataset.records),
        "n_valid": len(dataset.valid_records),
        "phases_applied": set(),
        "records": [],
    }

    for rec in dataset.valid_records:
        rec_prov = {
            "id": rec.source_id or rec.record_id,
            "provenance": list(rec.provenance),
        }
        provenance["records"].append(rec_prov)

        # Track phases
        for p in rec.provenance:
            if "validate" in p or "standardize" in p:
                provenance["phases_applied"].add("phase1_data")
            elif any(x in p for x in ["properties", "fingerprint", "filter", "descriptor"]):
                provenance["phases_applied"].add("phase1_analysis")
            elif any(x in p for x in ["qsar", "screen", "score", "sa_score", "drug_likeness"]):
                provenance["phases_applied"].add("phase2_screening")
            elif any(x in p for x in ["brics", "mutation", "scaffold", "ga_"]):
                provenance["phases_applied"].add("phase3_generation")
            elif any(x in p for x in ["conformer", "ligand_prep", "shape", "docking", "plif"]):
                provenance["phases_applied"].add("phase4_structure")

    provenance["phases_applied"] = sorted(provenance["phases_applied"])
    return provenance

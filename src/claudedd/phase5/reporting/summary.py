"""Multi-phase pipeline result aggregation and summarization.

Computes aggregate statistics across pipeline phases for
reporting and quality assessment.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from claudedd.core.models import MoleculeDataset

logger = logging.getLogger(__name__)


def summarize_dataset(dataset: MoleculeDataset) -> Dict[str, Any]:
    """Summarize a dataset with aggregate statistics.

    Args:
        dataset: MoleculeDataset to summarize.

    Returns:
        Dict with counts, property stats, filter results.
    """
    summary = dataset.summary()

    # Property statistics
    property_stats = {}
    numeric_props = [
        "MolWt", "LogP", "TPSA", "HBD", "HBA", "QED",
        "NumRotatableBonds", "RingCount", "FractionCSP3",
        "sa_score", "drug_likeness_score", "composite_score",
    ]

    for prop in numeric_props:
        values = [
            r.properties[prop]
            for r in dataset.valid_records
            if prop in r.properties and isinstance(r.properties[prop], (int, float))
        ]
        if values:
            property_stats[prop] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "count": len(values),
            }

    # Filter pass rates
    filter_stats = {}
    for filter_name in ["lipinski_pass", "pains_pass", "veber_pass", "brenk_pass"]:
        values = [
            r.properties.get(filter_name)
            for r in dataset.valid_records
            if filter_name in r.properties
        ]
        if values:
            n_pass = sum(1 for v in values if v is True)
            filter_stats[filter_name] = {
                "pass": n_pass,
                "fail": len(values) - n_pass,
                "rate": n_pass / len(values) if values else 0.0,
            }

    return {
        "name": dataset.name,
        "total_records": summary["total"],
        "valid_records": summary["valid"],
        "failed_records": summary["failed"],
        "property_stats": property_stats,
        "filter_stats": filter_stats,
    }


def summarize_generation(
    original: MoleculeDataset,
    generated: MoleculeDataset,
) -> Dict[str, Any]:
    """Summarize generation results vs original dataset.

    Args:
        original: Original seed dataset.
        generated: Generated molecules dataset.

    Returns:
        Dict with novelty, validity, diversity metrics.
    """
    original_smiles = {
        r.canonical_smiles
        for r in original.valid_records
        if r.canonical_smiles
    }
    generated_smiles = [
        r.canonical_smiles
        for r in generated.valid_records
        if r.canonical_smiles
    ]

    n_generated = len(generated.records)
    n_valid = len(generated.valid_records)
    unique_smiles = set(generated_smiles)
    novel_smiles = unique_smiles - original_smiles

    return {
        "n_generated": n_generated,
        "n_valid": n_valid,
        "validity_rate": n_valid / n_generated if n_generated > 0 else 0.0,
        "n_unique": len(unique_smiles),
        "uniqueness_rate": len(unique_smiles) / n_valid if n_valid > 0 else 0.0,
        "n_novel": len(novel_smiles),
        "novelty_rate": len(novel_smiles) / len(unique_smiles) if unique_smiles else 0.0,
    }


def summarize_scoring(dataset: MoleculeDataset) -> Dict[str, Any]:
    """Summarize scoring results.

    Args:
        dataset: Scored dataset.

    Returns:
        Dict with score distributions, candidate counts.
    """
    scores = [
        r.properties["composite_score"]
        for r in dataset.valid_records
        if "composite_score" in r.properties
    ]

    candidates = [
        r for r in dataset.valid_records
        if r.properties.get("candidate_flag", False)
    ]

    result = {
        "n_scored": len(scores),
        "n_candidates": len(candidates),
    }

    if scores:
        result["score_stats"] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores)),
        }
    else:
        result["score_stats"] = {}

    # Top molecules
    scored_recs = [
        (r, r.properties["composite_score"])
        for r in dataset.valid_records
        if "composite_score" in r.properties
    ]
    scored_recs.sort(key=lambda x: x[1], reverse=True)

    result["top_5"] = [
        {
            "id": r.source_id or r.record_id,
            "smiles": r.canonical_smiles or "",
            "score": round(s, 4),
        }
        for r, s in scored_recs[:5]
    ]

    return result


def create_pipeline_report(
    stages: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a full multi-phase pipeline report.

    Args:
        stages: Dict mapping stage name to stage results/summary.
        metadata: Optional metadata (workflow name, params, etc.).

    Returns:
        Complete pipeline report dict.
    """
    report = {
        "claudedd_version": _get_version(),
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {},
        "stages": stages,
    }

    return report


def _get_version() -> str:
    """Get ClaudeDD version."""
    try:
        from claudedd import __version__
        return __version__
    except ImportError:
        return "unknown"

"""Molecule ranking and candidate selection.

Ranks molecules by composite score and exports top candidates
for further evaluation.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from claudedd.core.exceptions import ScoringError
from claudedd.core.logging import get_logger
from claudedd.core.models import MoleculeDataset, MoleculeRecord

logger = get_logger("scoring.ranking")


def rank_molecules(
    dataset: MoleculeDataset,
    score_property: str = "composite_score",
    ascending: bool = False,
) -> List[Tuple[int, MoleculeRecord, float]]:
    """Rank molecules by a score property.

    Parameters
    ----------
    dataset : MoleculeDataset
        Dataset with scores in properties.
    score_property : str
        Property key to rank by.
    ascending : bool
        If True, lower scores rank higher.

    Returns
    -------
    list of (rank, record, score) tuples
        Sorted by score.
    """
    scored = []
    for i, rec in enumerate(dataset.valid_records):
        score = rec.properties.get(score_property)
        if score is not None:
            scored.append((i, rec, float(score)))

    if not scored:
        logger.warning(f"No records have '{score_property}' property")
        return []

    scored.sort(key=lambda x: x[2], reverse=not ascending)

    # Add rank to properties
    for rank, (idx, rec, score) in enumerate(scored, 1):
        rec.properties["rank"] = rank
        rec.properties["rank_by"] = score_property

    return scored


def get_top_candidates(
    dataset: MoleculeDataset,
    top_n: int = 50,
    score_property: str = "composite_score",
    ascending: bool = False,
) -> MoleculeDataset:
    """Get top-N candidates from a ranked dataset.

    Parameters
    ----------
    dataset : MoleculeDataset
        Dataset with scores.
    top_n : int
        Number of top candidates to return.
    score_property : str
        Property to rank by.
    ascending : bool
        If True, lower scores rank higher.

    Returns
    -------
    MoleculeDataset
        New dataset with only top-N records.
    """
    ranked = rank_molecules(dataset, score_property, ascending)

    top_records = [rec for _, rec, _ in ranked[:top_n]]

    result = MoleculeDataset(
        records=top_records,
        name=f"{dataset.name}_top{top_n}",
    )
    result._provenance = dataset._provenance + [
        f"ranked:top_{top_n}:{score_property}"
    ]

    logger.info(
        f"Selected top {len(top_records)} candidates by {score_property}"
    )
    return result


def flag_candidates(
    dataset: MoleculeDataset,
    criteria: Optional[Dict[str, Tuple[float, float]]] = None,
) -> MoleculeDataset:
    """Flag candidate molecules based on multi-criteria thresholds.

    Stores "candidate_flag" (True/False) in properties.

    Parameters
    ----------
    dataset : MoleculeDataset
        Dataset with scores.
    criteria : dict, optional
        Mapping of property name to (min_value, max_value) thresholds.
        Defaults to reasonable drug-likeness criteria.

    Returns
    -------
    MoleculeDataset
        Same dataset with candidate_flag added.
    """
    if criteria is None:
        criteria = {
            "composite_score": (0.5, float("inf")),
            "drug_likeness_score": (0.4, float("inf")),
            "sa_score": (float("-inf"), 6.0),
        }

    count = 0
    for rec in dataset.valid_records:
        passes = True
        for prop_name, (min_val, max_val) in criteria.items():
            val = rec.properties.get(prop_name)
            if val is not None:
                if float(val) < min_val or float(val) > max_val:
                    passes = False
                    break

        rec.properties["candidate_flag"] = passes
        if passes:
            count += 1

    logger.info(
        f"Flagged {count} candidates from {len(dataset.valid_records)} molecules"
    )
    return dataset


def export_ranked_results(
    dataset: MoleculeDataset,
    output_path: str,
    score_property: str = "composite_score",
    top_n: Optional[int] = None,
    include_properties: Optional[List[str]] = None,
) -> str:
    """Export ranked results to CSV.

    Parameters
    ----------
    dataset : MoleculeDataset
        Dataset with scores.
    output_path : str
        Output CSV file path.
    score_property : str
        Property to rank by.
    top_n : int, optional
        Only export top N results.
    include_properties : list of str, optional
        Specific properties to include (None = all).

    Returns
    -------
    str
        Path to output file.
    """
    ranked = rank_molecules(dataset, score_property)

    if top_n:
        ranked = ranked[:top_n]

    rows = []
    for rank, (idx, rec, score) in enumerate(ranked, 1):
        row = {
            "rank": rank,
            "record_id": rec.record_id,
            "source_id": rec.source_id,
            "smiles": rec.canonical_smiles,
            score_property: score,
        }

        if include_properties:
            for prop in include_properties:
                row[prop] = rec.properties.get(prop)
        else:
            # Include key scoring properties
            for key in [
                "predicted_activity", "drug_likeness_score",
                "sa_score", "sa_score_normalized",
                "composite_score", "candidate_flag",
                "QED", "MolWt", "LogP",
            ]:
                if key in rec.properties and key != score_property:
                    row[key] = rec.properties[key]

        rows.append(row)

    df = pd.DataFrame(rows)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Exported {len(rows)} ranked results to {output_path}")
    return output_path

"""Bioactivity data fetching, curation, and labeling.

Provides a composable pipeline for acquiring and preparing bioactivity
data from ChEMBL for QSAR modeling:

    fetch_chembl_bioactivity  ->  curate_bioactivity  ->  label_activity

Each function accepts and returns MoleculeDataset, so they can be
called independently or chained together.  Works equally well with
data fetched from the API or loaded from CSV via loaders.py.
"""

import csv
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem

from drugflow.core.constants import (
    BIOACTIVITY_ACTIVE_THRESHOLD_NM,
    BIOACTIVITY_CONFIDENCE_WEIGHTS,
    BIOACTIVITY_DEFAULT_TYPE,
    BIOACTIVITY_EXACT_RELATIONS,
    BIOACTIVITY_INACTIVE_THRESHOLD_NM,
    BIOACTIVITY_IQR_MULTIPLIER,
    BIOACTIVITY_TYPES,
    CHEMBL_ACTIVITY_FIELDS,
    CHEMBL_PAGE_SIZE,
    CHEMBL_REQUEST_DELAY,
)
from drugflow.core.exceptions import DatabaseError
from drugflow.core.logging import get_logger
from drugflow.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus

logger = get_logger("data.bioactivity")

CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"

# ── Unit conversion ──────────────────────────────────────────

_UNIT_CONVERSION_TO_NM: Dict[str, float] = {
    "M": 1e9,
    "mM": 1e6,
    "uM": 1e3,
    "nM": 1.0,
    "pM": 1e-3,
}


def _convert_to_nm(value: float, from_units: str) -> Optional[float]:
    """Convert an activity value to nanomolar.

    Parameters
    ----------
    value : float
        Activity value in original units.
    from_units : str
        Source units (M, mM, uM, nM, pM).

    Returns
    -------
    float or None
        Value in nM, or None if units are not recognized.
    """
    factor = _UNIT_CONVERSION_TO_NM.get(from_units)
    if factor is None:
        return None
    return value * factor


def compute_pic50(value_nm: float) -> float:
    """Convert an IC50 value in nM to pIC50.

    pIC50 = -log10(IC50 in M) = 9 - log10(IC50 in nM)

    Parameters
    ----------
    value_nm : float
        IC50 value in nanomolar.

    Returns
    -------
    float
        pIC50 value.  Returns 0.0 if value_nm <= 0.
    """
    if value_nm <= 0:
        return 0.0
    return 9.0 - math.log10(value_nm)


# ── Curation statistics ──────────────────────────────────────


@dataclass
class CurationStats:
    """Summary statistics from a curation run."""

    input_count: int = 0
    after_relation_filter: int = 0
    after_unit_filter: int = 0
    after_dedup: int = 0
    after_outlier_removal: int = 0
    output_count: int = 0
    duplicates_merged: int = 0
    outliers_removed: int = 0

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "input_count": self.input_count,
            "after_relation_filter": self.after_relation_filter,
            "after_unit_filter": self.after_unit_filter,
            "after_dedup": self.after_dedup,
            "after_outlier_removal": self.after_outlier_removal,
            "output_count": self.output_count,
            "duplicates_merged": self.duplicates_merged,
            "outliers_removed": self.outliers_removed,
        }


# ── Fetching ─────────────────────────────────────────────────


def _fetch_json(url: str, timeout: int = 30) -> dict:
    """Fetch JSON from a URL."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise DatabaseError(f"HTTP {e.code} error fetching {url}: {e.reason}")
    except urllib.error.URLError as e:
        raise DatabaseError(f"URL error fetching {url}: {e.reason}")


def _fetch_chembl_paginated(
    target_id: str,
    activity_type: str,
    max_results: int,
) -> List[dict]:
    """Fetch all activity records for one type with pagination.

    Handles ChEMBL's 1000-per-page limit, sleeping between requests.

    Parameters
    ----------
    target_id : str
        ChEMBL target ID.
    activity_type : str
        Single activity type (e.g., "IC50").
    max_results : int
        Maximum records to retrieve.

    Returns
    -------
    list of dict
        Raw activity records from ChEMBL JSON response.

    Raises
    ------
    DatabaseError
        If HTTP request fails.
    """
    all_activities: List[dict] = []
    offset = 0
    page_size = min(max_results, CHEMBL_PAGE_SIZE)

    while len(all_activities) < max_results:
        url = (
            f"{CHEMBL_API_BASE}/activity.json?"
            f"target_chembl_id={target_id}"
            f"&standard_type={activity_type}"
            f"&limit={page_size}"
            f"&offset={offset}"
        )

        logger.info(f"Fetching {activity_type} offset={offset} for {target_id}...")
        data = _fetch_json(url)
        activities = data.get("activities", [])

        if not activities:
            break

        all_activities.extend(activities)

        # Check if more pages exist
        next_url = data.get("page_meta", {}).get("next")
        if not next_url:
            break

        offset += page_size
        time.sleep(CHEMBL_REQUEST_DELAY)

    return all_activities[:max_results]


def _activity_to_record(
    activity: dict,
    target_id: str,
    index: int,
) -> MoleculeRecord:
    """Convert a single ChEMBL activity JSON dict to a MoleculeRecord.

    Parameters
    ----------
    activity : dict
        Raw ChEMBL activity record.
    target_id : str
        Target ChEMBL ID (for provenance).
    index : int
        Source index for ordering.

    Returns
    -------
    MoleculeRecord
    """
    smi = activity.get("canonical_smiles", "")
    mol = Chem.MolFromSmiles(smi) if smi else None

    rec = MoleculeRecord(
        mol=mol,
        smiles=smi,
        source_id=activity.get("molecule_chembl_id", ""),
        source_file=f"chembl:{target_id}",
        source_index=index,
        status=MoleculeStatus.RAW,
    )

    if mol is None:
        rec.add_error(f"Failed to parse SMILES: {smi}")

    # Store all relevant ChEMBL fields as metadata
    for field_name in CHEMBL_ACTIVITY_FIELDS:
        value = activity.get(field_name, "")
        rec.metadata[field_name] = value

    # Also store standard_value as a float for downstream processing
    raw_val = activity.get("standard_value")
    if raw_val is not None:
        try:
            rec.metadata["standard_value"] = float(raw_val)
        except (ValueError, TypeError):
            rec.metadata["standard_value"] = raw_val

    # Store pchembl_value as float if present
    raw_pchembl = activity.get("pchembl_value")
    if raw_pchembl is not None:
        try:
            rec.metadata["pchembl_value"] = float(raw_pchembl)
        except (ValueError, TypeError):
            rec.metadata["pchembl_value"] = raw_pchembl

    rec.add_provenance(f"fetched:chembl:{target_id}")
    return rec


def fetch_chembl_bioactivity(
    target_id: str,
    activity_types: Optional[List[str]] = None,
    max_results: int = 10000,
    save_raw: Optional[str] = None,
) -> MoleculeDataset:
    """Fetch bioactivity data from ChEMBL with pagination.

    Enhanced version of databases.fetch_chembl_by_target that supports
    paginated fetching, multiple activity types, and richer metadata.

    Parameters
    ----------
    target_id : str
        ChEMBL target identifier (e.g., "CHEMBL4860" for BCL-2).
    activity_types : list of str, optional
        Activity types to fetch. Defaults to ["IC50"].
        Valid: "IC50", "Ki", "Kd", "EC50".
    max_results : int
        Maximum total records to fetch. Default 10000.
    save_raw : str, optional
        Path to save raw fetched data as CSV backup.

    Returns
    -------
    MoleculeDataset
        Dataset with activity data in record.metadata.

    Raises
    ------
    DatabaseError
        If ChEMBL API request fails.
    ValueError
        If activity_types contains unsupported types.
    """
    if activity_types is None:
        activity_types = [BIOACTIVITY_DEFAULT_TYPE]

    # Validate activity types
    for atype in activity_types:
        if atype not in BIOACTIVITY_TYPES:
            raise ValueError(
                f"Unsupported activity type '{atype}'. "
                f"Valid types: {BIOACTIVITY_TYPES}"
            )

    records: List[MoleculeRecord] = []
    remaining = max_results

    for atype in activity_types:
        if remaining <= 0:
            break

        logger.info(f"Fetching {atype} data for target {target_id}...")
        activities = _fetch_chembl_paginated(target_id, atype, remaining)

        for i, act in enumerate(activities):
            rec = _activity_to_record(act, target_id, len(records) + i)
            records.append(rec)

        remaining -= len(activities)
        logger.info(f"  Got {len(activities)} {atype} records")

    dataset = MoleculeDataset(
        records=records,
        name=f"chembl_{target_id}_bioactivity",
    )

    # Optionally save raw backup as CSV
    if save_raw and records:
        _save_raw_csv(records, save_raw)
        logger.info(f"Raw data saved to {save_raw}")

    logger.info(
        f"Total: {len(records)} bioactivity records for {target_id} "
        f"(types: {activity_types})"
    )
    return dataset


def _save_raw_csv(records: List[MoleculeRecord], output_path: str) -> None:
    """Save raw fetched records to CSV backup."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fields = ["molecule_chembl_id", "canonical_smiles"] + [
        f for f in CHEMBL_ACTIVITY_FIELDS
        if f not in ("molecule_chembl_id",)
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = {"canonical_smiles": rec.smiles}
            row.update({k: rec.metadata.get(k, "") for k in fields})
            writer.writerow(row)


# ── Curation ─────────────────────────────────────────────────


def filter_by_relation(
    dataset: MoleculeDataset,
    allowed_relations: Optional[Set[str]] = None,
    relation_key: str = "standard_relation",
) -> MoleculeDataset:
    """Filter records by measurement relation type.

    Keeps only records whose metadata[relation_key] is in
    allowed_relations.

    Parameters
    ----------
    dataset : MoleculeDataset
        Input dataset.
    allowed_relations : set of str, optional
        Accepted relation values. Default {"="}.
    relation_key : str
        Metadata key containing the relation.

    Returns
    -------
    MoleculeDataset
        Filtered dataset containing only matching records.
    """
    if allowed_relations is None:
        allowed_relations = BIOACTIVITY_EXACT_RELATIONS

    kept = []
    for rec in dataset.records:
        relation = str(rec.metadata.get(relation_key, "")).strip()
        if relation in allowed_relations:
            kept.append(rec)
        else:
            rec.add_provenance("filter:relation")

    result = MoleculeDataset(
        records=kept,
        name=dataset.name,
        description=f"Relation-filtered ({allowed_relations})",
    )
    logger.info(
        f"Relation filter: {len(dataset)} -> {len(result)} "
        f"(kept relations: {allowed_relations})"
    )
    return result


def filter_and_normalize_units(
    dataset: MoleculeDataset,
    target_units: str = "nM",
    value_key: str = "standard_value",
    units_key: str = "standard_units",
) -> MoleculeDataset:
    """Filter to records with convertible units and normalize values.

    Converts activity values to the target unit (nM by default).
    Records with non-numeric values or unconvertible units are
    filtered out.

    Parameters
    ----------
    dataset : MoleculeDataset
        Input dataset.
    target_units : str
        Target unit. Default "nM".
    value_key : str
        Metadata key for activity value.
    units_key : str
        Metadata key for units.

    Returns
    -------
    MoleculeDataset
        Dataset with normalized values.
    """
    kept = []
    for rec in dataset.records:
        raw_value = rec.metadata.get(value_key)
        raw_units = str(rec.metadata.get(units_key, "")).strip()

        # Parse numeric value
        try:
            value = float(raw_value)
        except (ValueError, TypeError):
            rec.add_provenance("filter:non_numeric_value")
            continue

        # Convert units
        converted = _convert_to_nm(value, raw_units)
        if converted is None:
            rec.add_provenance(f"filter:unsupported_units:{raw_units}")
            continue

        # Update metadata with normalized value
        rec.metadata[value_key] = converted
        rec.metadata[units_key] = target_units
        kept.append(rec)

    result = MoleculeDataset(
        records=kept,
        name=dataset.name,
        description=f"Unit-normalized to {target_units}",
    )
    logger.info(
        f"Unit normalization: {len(dataset)} -> {len(result)} "
        f"(target: {target_units})"
    )
    return result


def deduplicate_molecules(
    dataset: MoleculeDataset,
    key: str = "canonical_smiles",
    value_key: str = "standard_value",
    aggregation: str = "median",
) -> MoleculeDataset:
    """Merge duplicate molecules by aggregating activity values.

    Groups records by canonical SMILES (or other key) and, for groups
    with multiple entries, computes the median (or mean) activity value.

    Parameters
    ----------
    dataset : MoleculeDataset
        Input dataset.
    key : str
        Grouping key: "canonical_smiles" or "inchikey".
    value_key : str
        Metadata key for activity value to aggregate.
    aggregation : str
        Aggregation method: "median" or "mean".

    Returns
    -------
    MoleculeDataset
        Deduplicated dataset with metadata["n_measurements"] set.
    """
    # Group records by key
    groups: Dict[str, List[MoleculeRecord]] = {}
    for rec in dataset.records:
        if key == "canonical_smiles":
            group_key = rec.smiles or ""
        elif key == "inchikey":
            group_key = rec.inchikey or rec.smiles or ""
        else:
            group_key = str(rec.metadata.get(key, rec.smiles or ""))

        if not group_key:
            continue
        groups.setdefault(group_key, []).append(rec)

    # Merge each group
    merged_records = []
    total_merged = 0

    for group_key, recs in groups.items():
        if len(recs) == 1:
            recs[0].metadata["n_measurements"] = 1
            merged_records.append(recs[0])
            continue

        # Collect activity values
        values = []
        for r in recs:
            try:
                values.append(float(r.metadata.get(value_key, 0)))
            except (ValueError, TypeError):
                pass

        # Choose representative record (highest confidence or first)
        best_rec = max(
            recs,
            key=lambda r: r.metadata.get("confidence_score", 0.0),
        )

        # Aggregate activity value
        if values:
            if aggregation == "median":
                agg_value = statistics.median(values)
            else:
                agg_value = statistics.mean(values)
            best_rec.metadata[value_key] = agg_value

        best_rec.metadata["n_measurements"] = len(recs)
        best_rec.add_provenance(
            f"curated:deduplicated:{aggregation}:n={len(recs)}"
        )
        merged_records.append(best_rec)
        total_merged += len(recs) - 1

    result = MoleculeDataset(
        records=merged_records,
        name=dataset.name,
        description=f"Deduplicated by {key} ({aggregation})",
    )
    logger.info(
        f"Deduplication: {len(dataset)} -> {len(result)} "
        f"({total_merged} duplicates merged)"
    )
    return result, total_merged


def remove_activity_outliers(
    dataset: MoleculeDataset,
    value_key: str = "standard_value",
    iqr_multiplier: float = BIOACTIVITY_IQR_MULTIPLIER,
) -> MoleculeDataset:
    """Remove activity value outliers using the IQR method.

    Computes Q1, Q3, and IQR on log10(activity_value).  Records
    outside [Q1 - k*IQR, Q3 + k*IQR] are filtered out.

    Parameters
    ----------
    dataset : MoleculeDataset
        Input dataset.
    value_key : str
        Metadata key for activity value.
    iqr_multiplier : float
        Multiplier for IQR bounds. Default 1.5.

    Returns
    -------
    MoleculeDataset
        Dataset with outliers removed.
    """
    # Collect log10 values
    log_values = []
    for rec in dataset.records:
        try:
            val = float(rec.metadata.get(value_key, 0))
            if val > 0:
                log_values.append(math.log10(val))
        except (ValueError, TypeError):
            pass

    if len(log_values) < 4:
        # Not enough data for IQR
        return dataset

    # Compute IQR bounds
    sorted_vals = sorted(log_values)
    n = len(sorted_vals)
    q1 = sorted_vals[n // 4]
    q3 = sorted_vals[3 * n // 4]
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr

    kept = []
    n_removed = 0
    for rec in dataset.records:
        try:
            val = float(rec.metadata.get(value_key, 0))
            if val > 0:
                log_val = math.log10(val)
                if lower_bound <= log_val <= upper_bound:
                    kept.append(rec)
                else:
                    rec.add_provenance("filter:outlier")
                    n_removed += 1
            else:
                rec.add_provenance("filter:outlier")
                n_removed += 1
        except (ValueError, TypeError):
            rec.add_provenance("filter:outlier")
            n_removed += 1

    result = MoleculeDataset(
        records=kept,
        name=dataset.name,
        description="Outliers removed (IQR)",
    )
    logger.info(
        f"Outlier removal: {len(dataset)} -> {len(result)} "
        f"({n_removed} outliers removed, "
        f"log10 bounds: [{lower_bound:.2f}, {upper_bound:.2f}])"
    )
    return result, n_removed


def compute_confidence_score(
    dataset: MoleculeDataset,
    relation_key: str = "standard_relation",
    pchembl_key: str = "pchembl_value",
    assay_type_key: str = "assay_type",
    units_key: str = "standard_units",
) -> MoleculeDataset:
    """Compute a data-quality confidence score for each record.

    Score is a weighted sum (0.0 to 1.0) of boolean factors:
    - exact_measurement: standard_relation == "="
    - pchembl_available: pchembl_value is not None/empty
    - trusted_assay: assay_type == "B" (binding)
    - unit_consistent: standard_units == "nM"

    Stores result in metadata["confidence_score"].

    Parameters
    ----------
    dataset : MoleculeDataset
        Input dataset.

    Returns
    -------
    MoleculeDataset
        Same dataset with confidence_score added to metadata.
    """
    weights = BIOACTIVITY_CONFIDENCE_WEIGHTS

    for rec in dataset.records:
        score = 0.0

        # Exact measurement
        relation = str(rec.metadata.get(relation_key, "")).strip()
        if relation == "=":
            score += weights["exact_measurement"]

        # pChEMBL available
        pchembl = rec.metadata.get(pchembl_key)
        if pchembl is not None and pchembl != "":
            try:
                float(pchembl)
                score += weights["pchembl_available"]
            except (ValueError, TypeError):
                pass

        # Trusted assay type (B = binding)
        assay_type = str(rec.metadata.get(assay_type_key, "")).strip()
        if assay_type == "B":
            score += weights["trusted_assay"]

        # Unit consistent
        units = str(rec.metadata.get(units_key, "")).strip()
        if units == "nM":
            score += weights["unit_consistent"]

        rec.metadata["confidence_score"] = round(score, 2)

    logger.info(f"Confidence scores computed for {len(dataset)} records")
    return dataset


def curate_bioactivity(
    dataset: MoleculeDataset,
    allowed_relations: Optional[Set[str]] = None,
    target_units: str = "nM",
    deduplicate: bool = True,
    dedup_key: str = "canonical_smiles",
    dedup_aggregation: str = "median",
    remove_outliers: bool = False,
    iqr_multiplier: float = BIOACTIVITY_IQR_MULTIPLIER,
    compute_confidence: bool = True,
    activity_value_key: str = "standard_value",
    activity_units_key: str = "standard_units",
    activity_relation_key: str = "standard_relation",
) -> Tuple[MoleculeDataset, CurationStats]:
    """Curate bioactivity data for QSAR readiness.

    Applies a sequence of filters and transformations:
    1. Filter by measurement relation (keep "=" by default)
    2. Filter and normalize units to target (nM)
    3. Deduplicate by canonical SMILES (median aggregation)
    4. Optionally remove activity value outliers (IQR)
    5. Optionally compute confidence scores

    Parameters
    ----------
    dataset : MoleculeDataset
        Input dataset with activity metadata.
    allowed_relations : set of str, optional
        Accepted standard_relation values. Default {"="}.
    target_units : str
        Target unit for normalization. Default "nM".
    deduplicate : bool
        Whether to merge duplicate molecules. Default True.
    dedup_key : str
        Key for deduplication. Default "canonical_smiles".
    dedup_aggregation : str
        Aggregation method: "median" or "mean". Default "median".
    remove_outliers : bool
        Whether to remove outliers. Default False.
    iqr_multiplier : float
        IQR multiplier for outlier bounds. Default 1.5.
    compute_confidence : bool
        Whether to compute confidence scores. Default True.
    activity_value_key : str
        Metadata key for the activity value.
    activity_units_key : str
        Metadata key for the activity units.
    activity_relation_key : str
        Metadata key for the activity relation.

    Returns
    -------
    dataset : MoleculeDataset
        Curated dataset.
    stats : CurationStats
        Statistics from each curation step.
    """
    stats = CurationStats()
    stats.input_count = len(dataset)

    # Step 1: Filter by relation
    current = filter_by_relation(
        dataset, allowed_relations, relation_key=activity_relation_key
    )
    stats.after_relation_filter = len(current)

    # Step 2: Normalize units
    current = filter_and_normalize_units(
        current, target_units,
        value_key=activity_value_key,
        units_key=activity_units_key,
    )
    stats.after_unit_filter = len(current)

    # Step 3: Compute confidence (before dedup so best-confidence
    # record is chosen as representative)
    if compute_confidence:
        current = compute_confidence_score(current)

    # Step 4: Deduplicate
    if deduplicate:
        current, n_merged = deduplicate_molecules(
            current,
            key=dedup_key,
            value_key=activity_value_key,
            aggregation=dedup_aggregation,
        )
        stats.duplicates_merged = n_merged
    stats.after_dedup = len(current)

    # Step 5: Remove outliers
    if remove_outliers:
        current, n_removed = remove_activity_outliers(
            current,
            value_key=activity_value_key,
            iqr_multiplier=iqr_multiplier,
        )
        stats.outliers_removed = n_removed
    stats.after_outlier_removal = len(current)

    stats.output_count = len(current)

    logger.info(
        f"Curation complete: {stats.input_count} -> {stats.output_count} "
        f"(merged: {stats.duplicates_merged}, outliers: {stats.outliers_removed})"
    )
    return current, stats


# ── Labeling ─────────────────────────────────────────────────


def label_activity(
    dataset: MoleculeDataset,
    mode: str = "binary",
    active_threshold: float = BIOACTIVITY_ACTIVE_THRESHOLD_NM,
    inactive_threshold: Optional[float] = None,
    value_key: str = "standard_value",
    units: str = "nM",
) -> MoleculeDataset:
    """Convert continuous activity values to categorical labels.

    Computes pIC50 and assigns activity classes for QSAR modeling.

    Binary mode:
        - activity_class = 1 if IC50 < active_threshold
        - activity_class = 0 if IC50 >= active_threshold

    Ternary mode:
        - activity_class = 1 (active) if IC50 < active_threshold
        - activity_class = 0 (intermediate)
        - activity_class = -1 (inactive) if IC50 >= inactive_threshold

    Stores in metadata: "pIC50", "activity_class", "activity_label".

    Parameters
    ----------
    dataset : MoleculeDataset
        Input dataset.
    mode : str
        "binary" or "ternary". Default "binary".
    active_threshold : float
        IC50 threshold in nM for active. Default 1000.0.
    inactive_threshold : float, optional
        IC50 threshold for inactive (ternary). Default 10000.0.
    value_key : str
        Metadata key for activity value.
    units : str
        Units of the value. Default "nM".

    Returns
    -------
    MoleculeDataset
        Dataset with labels in metadata.

    Raises
    ------
    ValueError
        If mode is invalid or ternary thresholds are inconsistent.
    """
    if mode not in ("binary", "ternary"):
        raise ValueError(f"Invalid mode '{mode}'. Use 'binary' or 'ternary'.")

    if inactive_threshold is None:
        inactive_threshold = BIOACTIVITY_INACTIVE_THRESHOLD_NM

    if mode == "ternary" and inactive_threshold <= active_threshold:
        raise ValueError(
            f"inactive_threshold ({inactive_threshold}) must be greater "
            f"than active_threshold ({active_threshold})."
        )

    for rec in dataset.records:
        raw_val = rec.metadata.get(value_key)
        try:
            value = float(raw_val)
        except (ValueError, TypeError):
            continue

        # Compute pIC50
        pic50 = compute_pic50(value)
        rec.metadata["pIC50"] = round(pic50, 4)

        # Assign class
        if mode == "binary":
            if value < active_threshold:
                rec.metadata["activity_class"] = 1
                rec.metadata["activity_label"] = "active"
            else:
                rec.metadata["activity_class"] = 0
                rec.metadata["activity_label"] = "inactive"
        else:  # ternary
            if value < active_threshold:
                rec.metadata["activity_class"] = 1
                rec.metadata["activity_label"] = "active"
            elif value >= inactive_threshold:
                rec.metadata["activity_class"] = -1
                rec.metadata["activity_label"] = "inactive"
            else:
                rec.metadata["activity_class"] = 0
                rec.metadata["activity_label"] = "intermediate"

        rec.add_provenance(f"labeled:{mode}")

    logger.info(f"Labeled {len(dataset)} records ({mode} mode)")
    return dataset


# ── Convenience pipeline ─────────────────────────────────────


def fetch_and_curate(
    target_id: str,
    activity_types: Optional[List[str]] = None,
    max_results: int = 10000,
    save_raw: Optional[str] = None,
    allowed_relations: Optional[Set[str]] = None,
    remove_outliers: bool = False,
    label_mode: str = "binary",
    active_threshold: float = BIOACTIVITY_ACTIVE_THRESHOLD_NM,
    inactive_threshold: Optional[float] = None,
) -> Tuple[MoleculeDataset, CurationStats]:
    """Full pipeline: fetch from ChEMBL, curate, and label.

    Convenience function that chains:
    fetch_chembl_bioactivity -> curate_bioactivity -> label_activity

    Parameters
    ----------
    target_id : str
        ChEMBL target ID.
    activity_types : list of str, optional
        Activity types to fetch.
    max_results : int
        Maximum records.
    save_raw : str, optional
        Path for raw CSV backup.
    allowed_relations : set of str, optional
        Accepted measurement relations.
    remove_outliers : bool
        Whether to remove outliers.
    label_mode : str
        "binary" or "ternary".
    active_threshold : float
        Active threshold in nM.
    inactive_threshold : float, optional
        Inactive threshold in nM (ternary only).

    Returns
    -------
    dataset : MoleculeDataset
        Fully curated and labeled dataset.
    stats : CurationStats
        Curation statistics.
    """
    # Fetch
    raw = fetch_chembl_bioactivity(
        target_id,
        activity_types=activity_types,
        max_results=max_results,
        save_raw=save_raw,
    )

    # Curate
    curated, stats = curate_bioactivity(
        raw,
        allowed_relations=allowed_relations,
        remove_outliers=remove_outliers,
    )

    # Label
    labeled = label_activity(
        curated,
        mode=label_mode,
        active_threshold=active_threshold,
        inactive_threshold=inactive_threshold,
    )

    return labeled, stats

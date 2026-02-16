"""QSAR data preparation: feature extraction, splitting, and scaling.

Extracts feature matrices from MoleculeDataset objects and provides
train/test splitting strategies (random and scaffold-based).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from claudedd.core.exceptions import ModelError
from claudedd.core.logging import get_logger
from claudedd.core.models import MoleculeDataset, MoleculeRecord
from claudedd.utils.chem import get_scaffold, mol_to_smiles

logger = get_logger("qsar.data_prep")


def extract_feature_matrix(
    dataset: MoleculeDataset,
    feature_source: str = "descriptors",
    feature_names: Optional[List[str]] = None,
    fp_type: Optional[str] = None,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Extract a feature matrix from a dataset.

    Parameters
    ----------
    dataset : MoleculeDataset
        Input dataset with computed descriptors/fingerprints.
    feature_source : str
        "descriptors" uses rec.descriptors, "fingerprints" uses rec.fingerprints.
    feature_names : list of str, optional
        Specific descriptor names to use. If None, uses all available.
    fp_type : str, optional
        Fingerprint key (required if feature_source="fingerprints").

    Returns
    -------
    X : np.ndarray
        Feature matrix (n_molecules, n_features).
    feature_names_out : list of str
        Feature column names.
    valid_indices : list of int
        Indices of records used (mapping back to dataset).

    Raises
    ------
    ModelError
        If no features can be extracted.
    """
    valid_records = dataset.valid_records

    if feature_source == "fingerprints":
        if fp_type is None:
            raise ModelError("fp_type must be specified for fingerprint features")
        return _extract_fingerprint_features(valid_records, fp_type)

    elif feature_source == "descriptors":
        return _extract_descriptor_features(valid_records, feature_names)

    else:
        raise ModelError(
            f"Unknown feature_source: '{feature_source}'. "
            f"Use 'descriptors' or 'fingerprints'."
        )


def _extract_descriptor_features(
    records: List[MoleculeRecord],
    feature_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Extract descriptor-based feature matrix."""
    # Determine feature names from first record that has descriptors
    if feature_names is None:
        for rec in records:
            if rec.descriptors:
                feature_names = sorted(rec.descriptors.keys())
                break
        if feature_names is None:
            raise ModelError(
                "No descriptors found. Compute descriptors first."
            )

    rows = []
    valid_indices = []
    for i, rec in enumerate(records):
        if not rec.descriptors:
            continue
        row = []
        valid = True
        for name in feature_names:
            val = rec.descriptors.get(name)
            if val is None:
                valid = False
                break
            row.append(float(val))
        if valid:
            rows.append(row)
            valid_indices.append(i)

    if not rows:
        raise ModelError("No valid descriptor rows could be extracted")

    X = np.array(rows, dtype=np.float64)

    # Handle NaN/inf
    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        logger.warning(
            f"Replacing {nan_mask.sum()} non-finite values with column means"
        )
        col_means = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
        for col_idx in range(X.shape[1]):
            bad = nan_mask[:, col_idx]
            X[bad, col_idx] = col_means[col_idx] if np.isfinite(col_means[col_idx]) else 0.0

    return X, list(feature_names), valid_indices


def _extract_fingerprint_features(
    records: List[MoleculeRecord],
    fp_type: str,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Extract fingerprint-based feature matrix."""
    rows = []
    valid_indices = []
    for i, rec in enumerate(records):
        if fp_type in rec.fingerprints:
            rows.append(rec.fingerprints[fp_type].astype(np.float64))
            valid_indices.append(i)

    if not rows:
        raise ModelError(
            f"No fingerprints of type '{fp_type}' found. "
            f"Compute fingerprints first."
        )

    X = np.array(rows, dtype=np.float64)
    n_bits = X.shape[1]
    feature_names = [f"{fp_type}_bit_{i}" for i in range(n_bits)]

    return X, feature_names, valid_indices


def extract_labels(
    dataset: MoleculeDataset,
    activity_col: str,
    valid_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """Extract activity labels from dataset.

    Parameters
    ----------
    dataset : MoleculeDataset
        Input dataset.
    activity_col : str
        Property or metadata key containing activity values.
    valid_indices : list of int, optional
        If provided, only extract labels for these record indices
        (as returned by extract_feature_matrix).

    Returns
    -------
    np.ndarray
        Activity values.

    Raises
    ------
    ModelError
        If activity column not found.
    """
    records = dataset.valid_records
    if valid_indices is not None:
        records = [records[i] for i in valid_indices]

    labels = []
    for rec in records:
        # Check properties first, then metadata
        val = rec.properties.get(activity_col)
        if val is None:
            val = rec.metadata.get(activity_col)
        if val is None:
            raise ModelError(
                f"Activity column '{activity_col}' not found in record "
                f"{rec.record_id}. Available properties: "
                f"{list(rec.properties.keys())}, "
                f"metadata: {list(rec.metadata.keys())}"
            )
        labels.append(float(val))

    return np.array(labels, dtype=np.float64)


def random_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random train/test split.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    test_size : float
        Fraction for test set.
    random_state : int
        Random seed.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def scaffold_split(
    dataset: MoleculeDataset,
    X: np.ndarray,
    y: np.ndarray,
    valid_indices: List[int],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scaffold-based train/test split to prevent data leakage.

    Molecules with the same Murcko scaffold are kept in the same split.

    Parameters
    ----------
    dataset : MoleculeDataset
        Original dataset (for scaffold computation).
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    valid_indices : list of int
        Indices mapping X rows to dataset.valid_records.
    test_size : float
        Approximate fraction for test set.
    random_state : int
        Random seed.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    records = dataset.valid_records

    # Compute scaffolds
    scaffold_to_indices: Dict[str, List[int]] = {}
    for row_idx, rec_idx in enumerate(valid_indices):
        rec = records[rec_idx]
        if rec.mol is not None:
            try:
                scaffold = get_scaffold(rec.mol)
                scaffold_smi = mol_to_smiles(scaffold)
            except Exception:
                scaffold_smi = f"_no_scaffold_{row_idx}"
        else:
            scaffold_smi = f"_no_scaffold_{row_idx}"

        if scaffold_smi not in scaffold_to_indices:
            scaffold_to_indices[scaffold_smi] = []
        scaffold_to_indices[scaffold_smi].append(row_idx)

    # Sort scaffolds by size (largest first) for balanced splitting
    sorted_scaffolds = sorted(
        scaffold_to_indices.items(), key=lambda x: len(x[1]), reverse=True
    )

    rng = np.random.RandomState(random_state)
    # Shuffle scaffolds of same size
    rng.shuffle(sorted_scaffolds)

    n_test = int(len(X) * test_size)
    test_indices = []
    train_indices = []

    for _, indices in sorted_scaffolds:
        if len(test_indices) < n_test:
            test_indices.extend(indices)
        else:
            train_indices.extend(indices)

    # Ensure we have both sets
    if not test_indices or not train_indices:
        logger.warning(
            "Scaffold split produced empty set, falling back to random split"
        )
        return random_split(X, y, test_size, random_state)

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def scale_features(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], "StandardScaler"]:
    """Scale features using StandardScaler.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    X_test : np.ndarray, optional
        Test feature matrix to transform using training statistics.

    Returns
    -------
    X_train_scaled : np.ndarray
    X_test_scaled : np.ndarray or None
    scaler : StandardScaler
        Fitted scaler for future use.
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

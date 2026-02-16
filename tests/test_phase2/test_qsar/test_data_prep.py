"""Tests for QSAR data preparation."""

import numpy as np
import pytest

from claudedd.core.exceptions import ModelError
from claudedd.phase2.qsar.data_prep import (
    extract_feature_matrix,
    extract_labels,
    random_split,
    scaffold_split,
    scale_features,
)


def test_extract_descriptor_features(computed_dataset):
    """Extract descriptor feature matrix."""
    X, names, indices = extract_feature_matrix(
        computed_dataset, feature_source="descriptors",
    )
    assert X.shape[0] == len(computed_dataset.valid_records)
    assert X.shape[1] > 0
    assert len(names) == X.shape[1]
    assert len(indices) == X.shape[0]


def test_extract_fingerprint_features(computed_dataset):
    """Extract fingerprint feature matrix."""
    X, names, indices = extract_feature_matrix(
        computed_dataset,
        feature_source="fingerprints",
        fp_type="morgan_r2_2048",
    )
    assert X.shape[0] == len(computed_dataset.valid_records)
    assert X.shape[1] == 2048


def test_extract_features_no_fp_type(computed_dataset):
    """Fingerprint source without fp_type raises ModelError."""
    with pytest.raises(ModelError, match="fp_type must be specified"):
        extract_feature_matrix(
            computed_dataset, feature_source="fingerprints",
        )


def test_extract_labels(computed_dataset):
    """Extract activity labels from metadata."""
    y = extract_labels(computed_dataset, "activity")
    assert len(y) == len(computed_dataset.valid_records)
    assert y.dtype == np.float64


def test_extract_labels_missing_col(computed_dataset):
    """Missing activity column raises ModelError."""
    with pytest.raises(ModelError, match="not found"):
        extract_labels(computed_dataset, "nonexistent_column")


def test_random_split():
    """Random split produces correct sizes."""
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    X_train, X_test, y_train, y_test = random_split(X, y, test_size=0.2)
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_scaffold_split(computed_dataset):
    """Scaffold split produces non-empty train and test sets."""
    X, names, indices = extract_feature_matrix(
        computed_dataset, feature_source="descriptors",
    )
    y = extract_labels(computed_dataset, "activity")

    X_train, X_test, y_train, y_test = scaffold_split(
        computed_dataset, X, y, indices, test_size=0.4,
    )
    assert len(y_train) > 0
    assert len(y_test) > 0
    assert len(y_train) + len(y_test) == len(y)


def test_scale_features():
    """StandardScaler normalizes features."""
    X_train = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    X_test = np.array([[2.5, 250.0]])

    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    # Scaled training set should have mean ~0 and std ~1
    assert abs(X_train_s.mean()) < 0.1
    assert X_test_s is not None
    assert scaler is not None


def test_scale_features_no_test():
    """Scale without test set."""
    X_train = np.random.rand(20, 5)
    X_train_s, X_test_s, scaler = scale_features(X_train)
    assert X_test_s is None
    assert scaler is not None

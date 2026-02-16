"""Tests for uncertainty estimation."""

import numpy as np
import pytest

from drugflow.core.exceptions import ModelError
from drugflow.phase3.active_learning.uncertainty import (
    compute_ensemble_uncertainty,
    compute_prediction_spread,
    rank_by_uncertainty,
)


def test_ensemble_uncertainty_rf(trained_rf_model, computed_dataset):
    """RF model produces uncertainty estimates."""
    from drugflow.phase2.qsar.data_prep import extract_feature_matrix

    X, _, _ = extract_feature_matrix(computed_dataset, feature_source="descriptors")
    uncertainties = compute_ensemble_uncertainty(trained_rf_model, X)
    assert len(uncertainties) == X.shape[0]
    assert all(u >= 0.0 for u in uncertainties)


def test_prediction_spread_rf(trained_rf_model, computed_dataset):
    """RF model produces prediction spread."""
    from drugflow.phase2.qsar.data_prep import extract_feature_matrix

    X, _, _ = extract_feature_matrix(computed_dataset, feature_source="descriptors")
    spread = compute_prediction_spread(trained_rf_model, X)
    assert len(spread) == X.shape[0]
    assert all(s >= 0.0 for s in spread)


def test_rank_by_uncertainty(trained_rf_model, computed_dataset):
    """Rank molecules by uncertainty."""
    ranked = rank_by_uncertainty(trained_rf_model, computed_dataset)
    assert len(ranked) > 0
    # Should be sorted descending by uncertainty
    for i in range(len(ranked) - 1):
        assert ranked[i][1] >= ranked[i + 1][1]


def test_uncertainty_unsupported_model(computed_dataset):
    """Non-ensemble model raises ModelError."""
    from drugflow.phase2.qsar.data_prep import extract_feature_matrix

    X, _, _ = extract_feature_matrix(computed_dataset, feature_source="descriptors")

    # Create a simple mock model without estimators_
    class MockModel:
        def predict(self, X):
            return np.zeros(X.shape[0])

    with pytest.raises(ModelError, match="does not support"):
        compute_ensemble_uncertainty(MockModel(), X)

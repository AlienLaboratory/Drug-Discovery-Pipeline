"""Tests for QSAR model evaluation."""

import numpy as np
import pytest

from drugflow.phase2.qsar.evaluation import (
    evaluate_model,
    cross_validate,
    y_randomization_test,
    compute_applicability_domain,
)


@pytest.fixture
def regression_data():
    np.random.seed(42)
    X = np.random.rand(50, 5)
    y = X[:, 0] * 2 + X[:, 1] + np.random.rand(50) * 0.1
    return X, y


@pytest.fixture
def classification_data():
    np.random.seed(42)
    X = np.random.rand(50, 5)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(float)
    return X, y


def test_evaluate_regression():
    """Regression evaluation returns expected metrics."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    metrics = evaluate_model(y_true, y_pred, task="regression")
    assert "r2" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "pearson_r" in metrics
    assert metrics["r2"] > 0.9


def test_evaluate_classification():
    """Classification evaluation returns expected metrics."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 0])
    metrics = evaluate_model(y_true, y_pred, task="classification")
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert metrics["accuracy"] == 0.8


def test_cross_validate_regression(regression_data):
    """Cross-validation produces fold metrics."""
    X, y = regression_data
    results = cross_validate(X, y, n_folds=3, task="regression")
    assert results["n_folds"] == 3
    assert len(results["fold_metrics"]) == 3
    assert "mean_metrics" in results
    assert "r2" in results["mean_metrics"]


def test_cross_validate_classification(classification_data):
    """Classification cross-validation works."""
    X, y = classification_data
    results = cross_validate(X, y, n_folds=3, task="classification")
    assert "accuracy" in results["mean_metrics"]


def test_y_randomization(regression_data):
    """Y-randomization test validates model."""
    X, y = regression_data
    results = y_randomization_test(X, y, n_iterations=5)
    assert "original_metrics" in results
    assert "mean_randomized_metrics" in results
    assert "is_valid" in results
    # Original should be better than random
    assert results["original_score"] > results["mean_randomized_score"]


def test_applicability_domain_leverage(regression_data):
    """Leverage AD computes for query molecules."""
    X, _ = regression_data
    X_train = X[:40]
    X_query = X[40:]
    result = compute_applicability_domain(X_train, X_query, method="leverage")
    assert "leverages" in result
    assert "threshold" in result
    assert "in_domain" in result
    assert "fraction_in_domain" in result
    assert len(result["leverages"]) == 10


def test_applicability_domain_distance(regression_data):
    """Distance AD computes for query molecules."""
    X, _ = regression_data
    X_train = X[:40]
    X_query = X[40:]
    result = compute_applicability_domain(X_train, X_query, method="distance")
    assert result["method"] == "distance"
    assert len(result["distances"]) == 10

"""Tests for QSAR model training."""

import numpy as np
import pytest

from claudedd.core.exceptions import ModelError
from claudedd.phase2.qsar.models import (
    QSARModel,
    train_model,
    _check_xgboost_available,
)


@pytest.fixture
def regression_data():
    """Simple regression dataset."""
    np.random.seed(42)
    X = np.random.rand(50, 5)
    y = X[:, 0] * 2 + X[:, 1] + np.random.rand(50) * 0.1
    return X, y


@pytest.fixture
def classification_data():
    """Simple binary classification dataset."""
    np.random.seed(42)
    X = np.random.rand(50, 5)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(float)
    return X, y


def test_train_random_forest_regression(regression_data):
    """Train RF regressor."""
    X, y = regression_data
    model = train_model(X, y, model_type="random_forest", task="regression")
    assert model.model_type == "random_forest"
    assert model.task == "regression"
    assert model.n_features == 5
    assert model.training_metrics["r2"] > 0.5


def test_train_gradient_boosting_regression(regression_data):
    """Train GBT regressor."""
    X, y = regression_data
    model = train_model(X, y, model_type="gradient_boosting", task="regression")
    assert model.model_type == "gradient_boosting"
    assert model.training_metrics["r2"] > 0.5


def test_train_random_forest_classification(classification_data):
    """Train RF classifier."""
    X, y = classification_data
    model = train_model(X, y, model_type="random_forest", task="classification")
    assert model.is_classifier
    assert model.training_metrics["accuracy"] > 0.5


def test_model_predict(regression_data):
    """Model makes predictions."""
    X, y = regression_data
    model = train_model(X, y)
    predictions = model.predict(X[:5])
    assert len(predictions) == 5


def test_model_predict_proba(classification_data):
    """Classifier provides probabilities."""
    X, y = classification_data
    model = train_model(X, y, task="classification")
    probas = model.predict_proba(X[:5])
    assert probas.shape == (5, 2)
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_model_predict_proba_regressor_fails(regression_data):
    """Regressor cannot return probabilities."""
    X, y = regression_data
    model = train_model(X, y, task="regression")
    with pytest.raises(ModelError, match="predict_proba only available"):
        model.predict_proba(X[:5])


def test_model_feature_importances(regression_data):
    """Feature importances are available."""
    X, y = regression_data
    model = train_model(
        X, y, feature_names=["f1", "f2", "f3", "f4", "f5"],
    )
    importances = model.get_feature_importances()
    assert importances is not None
    assert "f1" in importances
    assert sum(importances.values()) == pytest.approx(1.0, abs=0.01)


def test_model_summary(regression_data):
    """Model summary contains expected keys."""
    X, y = regression_data
    model = train_model(X, y, feature_names=["f1", "f2", "f3", "f4", "f5"])
    summary = model.summary()
    assert "model_type" in summary
    assert "task" in summary
    assert "n_features" in summary
    assert "training_metrics" in summary


def test_train_invalid_model_type(regression_data):
    """Invalid model type raises ModelError."""
    X, y = regression_data
    with pytest.raises(ModelError, match="Unknown model_type"):
        train_model(X, y, model_type="invalid_model")


def test_train_invalid_task(regression_data):
    """Invalid task raises ModelError."""
    X, y = regression_data
    with pytest.raises(ModelError, match="Unknown task"):
        train_model(X, y, task="invalid_task")

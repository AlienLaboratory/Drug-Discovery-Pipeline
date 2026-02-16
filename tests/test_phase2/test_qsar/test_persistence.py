"""Tests for QSAR model persistence."""

import numpy as np
import pytest

from claudedd.core.exceptions import ModelError
from claudedd.phase2.qsar.models import train_model
from claudedd.phase2.qsar.persistence import save_model, load_model


@pytest.fixture
def trained_model():
    np.random.seed(42)
    X = np.random.rand(30, 3)
    y = X[:, 0] + X[:, 1] * 2
    return train_model(
        X, y, feature_names=["a", "b", "c"], dataset_name="test_data",
    )


def test_save_and_load(trained_model, tmp_path):
    """Save and load preserves model."""
    path = str(tmp_path / "model.joblib")
    save_model(trained_model, path)
    loaded = load_model(path)

    assert loaded.model_type == trained_model.model_type
    assert loaded.task == trained_model.task
    assert loaded.feature_names == trained_model.feature_names
    assert loaded.n_features == trained_model.n_features


def test_load_predictions_match(trained_model, tmp_path):
    """Loaded model produces same predictions."""
    path = str(tmp_path / "model.joblib")
    save_model(trained_model, path)
    loaded = load_model(path)

    X_test = np.random.rand(5, 3)
    orig_pred = trained_model.predict(X_test)
    loaded_pred = loaded.predict(X_test)
    np.testing.assert_array_almost_equal(orig_pred, loaded_pred)


def test_load_nonexistent():
    """Loading nonexistent file raises ModelError."""
    with pytest.raises(ModelError, match="not found"):
        load_model("/nonexistent/path/model.joblib")


def test_save_creates_directory(trained_model, tmp_path):
    """Save creates parent directories."""
    path = str(tmp_path / "subdir" / "nested" / "model.joblib")
    saved_path = save_model(trained_model, path)
    assert saved_path.endswith("model.joblib")
    loaded = load_model(path)
    assert loaded.model_type == "random_forest"

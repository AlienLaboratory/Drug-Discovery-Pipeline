"""Tests for QSAR prediction."""

import numpy as np
import pytest

from claudedd.core.exceptions import ModelError
from claudedd.phase2.qsar.data_prep import extract_feature_matrix, extract_labels
from claudedd.phase2.qsar.models import train_model
from claudedd.phase2.qsar.prediction import predict_dataset, predict_single


@pytest.fixture
def trained_descriptor_model(computed_dataset):
    """Train a model on computed dataset descriptors."""
    X, names, indices = extract_feature_matrix(
        computed_dataset, feature_source="descriptors",
    )
    y = extract_labels(computed_dataset, "activity", indices)
    return train_model(X, y, feature_names=names, task="regression")


@pytest.fixture
def trained_fp_model(computed_dataset):
    """Train a model on computed dataset fingerprints."""
    X, names, indices = extract_feature_matrix(
        computed_dataset,
        feature_source="fingerprints",
        fp_type="morgan_r2_2048",
    )
    y = extract_labels(computed_dataset, "activity", indices)
    return train_model(X, y, feature_names=names, task="regression")


def test_predict_dataset_descriptors(trained_descriptor_model, computed_dataset):
    """Predict with descriptor-based model."""
    result = predict_dataset(
        trained_descriptor_model, computed_dataset,
        feature_source="descriptors",
    )
    for rec in result.valid_records:
        assert "predicted_activity" in rec.properties


def test_predict_dataset_fingerprints(trained_fp_model, computed_dataset):
    """Predict with fingerprint-based model."""
    result = predict_dataset(
        trained_fp_model, computed_dataset,
        feature_source="fingerprints", fp_type="morgan_r2_2048",
    )
    for rec in result.valid_records:
        assert "predicted_activity" in rec.properties


def test_predict_single_descriptor(trained_descriptor_model, computed_dataset):
    """Predict single molecule."""
    rec = computed_dataset.valid_records[0]
    pred = predict_single(
        trained_descriptor_model, rec, feature_source="descriptors",
    )
    assert isinstance(pred, float)


def test_predict_single_no_descriptors(trained_descriptor_model, aspirin_record):
    """Predict single without descriptors raises ModelError."""
    with pytest.raises(ModelError, match="no descriptors"):
        predict_single(
            trained_descriptor_model, aspirin_record,
            feature_source="descriptors",
        )

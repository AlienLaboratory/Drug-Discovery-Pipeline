"""QSAR prediction: apply trained models to new molecules.

Extracts features from a MoleculeDataset and stores predictions
in rec.properties for downstream scoring and ranking.
"""

from typing import Dict, List, Optional

import numpy as np

from drugflow.core.exceptions import ModelError
from drugflow.core.logging import get_logger, progress_bar
from drugflow.core.models import MoleculeDataset, MoleculeRecord
from drugflow.phase2.qsar.data_prep import extract_feature_matrix
from drugflow.phase2.qsar.models import QSARModel

logger = get_logger("qsar.prediction")


def predict_dataset(
    model: QSARModel,
    dataset: MoleculeDataset,
    feature_source: str = "descriptors",
    fp_type: Optional[str] = None,
    output_property: str = "predicted_activity",
) -> MoleculeDataset:
    """Apply a trained model to predict activity for a dataset.

    Stores predictions in rec.properties[output_property].

    Parameters
    ----------
    model : QSARModel
        Trained QSAR model.
    dataset : MoleculeDataset
        Input dataset (must have features matching model).
    feature_source : str
        "descriptors" or "fingerprints".
    fp_type : str, optional
        Fingerprint type (required if feature_source="fingerprints").
    output_property : str
        Property key for storing predictions.

    Returns
    -------
    MoleculeDataset
        Dataset with predictions added to properties.

    Raises
    ------
    ModelError
        If feature extraction fails or feature mismatch.
    """
    # Extract features
    try:
        X, feature_names, valid_indices = extract_feature_matrix(
            dataset,
            feature_source=feature_source,
            feature_names=model.feature_names if feature_source == "descriptors" else None,
            fp_type=fp_type,
        )
    except ModelError as e:
        raise ModelError(f"Cannot extract features for prediction: {e}")

    # Validate feature alignment
    if X.shape[1] != model.n_features:
        raise ModelError(
            f"Feature mismatch: model expects {model.n_features} features, "
            f"got {X.shape[1]}"
        )

    # Make predictions
    logger.info(
        f"Predicting {len(valid_indices)} molecules with {model.model_type}"
    )
    predictions = model.predict(X)

    # Store predictions in records
    valid_records = dataset.valid_records
    for row_idx, rec_idx in enumerate(valid_indices):
        rec = valid_records[rec_idx]
        rec.properties[output_property] = float(predictions[row_idx])

        # For classifiers, also store probabilities
        if model.is_classifier:
            try:
                probas = model.predict_proba(X[row_idx:row_idx+1])
                rec.properties[f"{output_property}_proba"] = probas[0].tolist()
            except Exception:
                pass

        rec.add_provenance(f"qsar:predicted:{model.model_type}")

    # Mark records without predictions
    predicted_set = set(valid_indices)
    for i, rec in enumerate(valid_records):
        if i not in predicted_set and output_property not in rec.properties:
            rec.properties[output_property] = None
            rec.properties[f"{output_property}_note"] = "no_features"

    logger.info(
        f"Predictions complete: {len(valid_indices)} molecules predicted"
    )
    return dataset


def predict_single(
    model: QSARModel,
    record: MoleculeRecord,
    feature_source: str = "descriptors",
    fp_type: Optional[str] = None,
) -> float:
    """Predict activity for a single molecule.

    Parameters
    ----------
    model : QSARModel
        Trained model.
    record : MoleculeRecord
        Molecule with computed features.
    feature_source : str
        "descriptors" or "fingerprints".
    fp_type : str, optional
        Fingerprint type.

    Returns
    -------
    float
        Predicted activity value.

    Raises
    ------
    ModelError
        If prediction fails.
    """
    if feature_source == "descriptors":
        if not record.descriptors:
            raise ModelError("Record has no descriptors computed")
        row = []
        for name in model.feature_names:
            val = record.descriptors.get(name)
            if val is None:
                raise ModelError(f"Missing descriptor: {name}")
            row.append(float(val))
        X = np.array([row], dtype=np.float64)

    elif feature_source == "fingerprints":
        if fp_type is None:
            raise ModelError("fp_type required for fingerprint features")
        if fp_type not in record.fingerprints:
            raise ModelError(f"Fingerprint '{fp_type}' not found")
        X = record.fingerprints[fp_type].reshape(1, -1).astype(np.float64)

    else:
        raise ModelError(f"Unknown feature_source: '{feature_source}'")

    if X.shape[1] != model.n_features:
        raise ModelError(
            f"Feature mismatch: model expects {model.n_features}, "
            f"got {X.shape[1]}"
        )

    predictions = model.predict(X)
    return float(predictions[0])

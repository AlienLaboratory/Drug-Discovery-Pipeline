"""Uncertainty estimation for active learning.

Computes prediction uncertainty from ensemble models (Random Forest,
Gradient Boosting) by measuring disagreement across individual trees.
"""

import logging
from typing import Any, List, Optional, Tuple

import numpy as np

from drugflow.core.exceptions import ModelError
from drugflow.core.models import MoleculeDataset

logger = logging.getLogger(__name__)


def compute_ensemble_uncertainty(
    model: Any,
    X: np.ndarray,
) -> np.ndarray:
    """Compute prediction uncertainty as standard deviation across ensemble members.

    For Random Forest: uses individual tree predictions.
    For Gradient Boosting: uses staged predictions (partial ensembles).

    Args:
        model: QSARModel (from Phase 2) with a tree-based estimator.
        X: Feature matrix of shape (n_samples, n_features).

    Returns:
        Array of uncertainty values, shape (n_samples,).

    Raises:
        ModelError: If model doesn't support ensemble uncertainty.
    """
    estimator = model.estimator if hasattr(model, "estimator") else model

    # Apply scaler if present
    if hasattr(model, "scaler") and model.scaler is not None:
        X = model.scaler.transform(X)

    # Random Forest: access individual trees
    if hasattr(estimator, "estimators_"):
        predictions = np.array([
            tree.predict(X) for tree in estimator.estimators_
        ])  # shape: (n_trees, n_samples)
        return np.std(predictions, axis=0)

    # Gradient Boosting: use staged_predict for partial ensembles
    if hasattr(estimator, "staged_predict"):
        staged = list(estimator.staged_predict(X))
        if len(staged) >= 2:
            # Use last N stages to estimate uncertainty
            n_stages = min(10, len(staged))
            last_stages = np.array(staged[-n_stages:])
            return np.std(last_stages, axis=0)

    raise ModelError(
        "Model does not support ensemble uncertainty. "
        "Use Random Forest or Gradient Boosting."
    )


def compute_prediction_spread(
    model: Any,
    X: np.ndarray,
) -> np.ndarray:
    """Compute prediction spread (max - min) across ensemble members.

    A broader spread indicates higher model disagreement and uncertainty.

    Args:
        model: QSARModel with tree-based estimator.
        X: Feature matrix.

    Returns:
        Array of spread values, shape (n_samples,).
    """
    estimator = model.estimator if hasattr(model, "estimator") else model

    if hasattr(model, "scaler") and model.scaler is not None:
        X = model.scaler.transform(X)

    if hasattr(estimator, "estimators_"):
        predictions = np.array([
            tree.predict(X) for tree in estimator.estimators_
        ])
        return np.max(predictions, axis=0) - np.min(predictions, axis=0)

    raise ModelError("Model does not support prediction spread estimation")


def rank_by_uncertainty(
    model: Any,
    dataset: MoleculeDataset,
    feature_source: str = "descriptors",
    fp_type: Optional[str] = None,
) -> List[Tuple[int, float]]:
    """Rank dataset molecules by prediction uncertainty (highest first).

    Args:
        model: Trained QSARModel.
        dataset: Dataset with computed features.
        feature_source: "descriptors" or "fingerprints".
        fp_type: Fingerprint type if using fingerprints.

    Returns:
        List of (record_index, uncertainty) tuples, sorted descending.
    """
    from drugflow.phase2.qsar.data_prep import extract_feature_matrix

    X, _, indices = extract_feature_matrix(
        dataset, feature_source=feature_source, fp_type=fp_type,
    )

    if X.shape[0] == 0:
        return []

    uncertainties = compute_ensemble_uncertainty(model, X)

    # Map back to dataset indices with uncertainty
    ranked = [
        (int(indices[i]), float(uncertainties[i]))
        for i in range(len(indices))
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked

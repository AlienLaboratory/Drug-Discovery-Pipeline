"""Acquisition functions for active learning.

Provides strategies for selecting the most informative molecules from
a pool for experimental testing: greedy, uncertainty, UCB, diversity,
and balanced (exploration-exploitation trade-off).
"""

import logging
from typing import Any, List, Optional

import numpy as np

from drugflow.core.constants import AL_DEFAULT_BATCH_SIZE, AL_DEFAULT_KAPPA
from drugflow.core.exceptions import GenerationError
from drugflow.core.models import MoleculeDataset, MoleculeRecord
from drugflow.phase3.active_learning.diversity import maxmin_diversity_pick
from drugflow.phase3.active_learning.uncertainty import compute_ensemble_uncertainty

logger = logging.getLogger(__name__)


def _extract_features(
    dataset: MoleculeDataset,
    feature_source: str = "descriptors",
    fp_type: Optional[str] = None,
):
    """Extract feature matrix from dataset using Phase 2 tools."""
    from drugflow.phase2.qsar.data_prep import extract_feature_matrix

    return extract_feature_matrix(
        dataset, feature_source=feature_source, fp_type=fp_type,
    )


def _get_fps_matrix(
    dataset: MoleculeDataset,
    fp_type: str = "morgan_r2_2048",
) -> np.ndarray:
    """Extract fingerprint matrix for diversity calculations."""
    fps = []
    for rec in dataset.valid_records:
        if fp_type in rec.fingerprints:
            fps.append(rec.fingerprints[fp_type])
        else:
            fps.append(np.zeros(2048))
    return np.array(fps) if fps else np.empty((0, 2048))


def _select_by_indices(
    dataset: MoleculeDataset,
    indices: List[int],
    strategy_name: str,
) -> MoleculeDataset:
    """Create a new dataset from selected valid record indices."""
    valid = dataset.valid_records
    records = []
    for idx in indices:
        if idx < len(valid):
            rec = valid[idx]
            rec.metadata["al_strategy"] = strategy_name
            records.append(rec)
    return MoleculeDataset(
        records=records,
        name=f"al_{strategy_name}_{len(records)}",
    )


def greedy_acquisition(
    dataset: MoleculeDataset,
    model: Any,
    batch_size: int = AL_DEFAULT_BATCH_SIZE,
    feature_source: str = "descriptors",
    fp_type: Optional[str] = None,
) -> MoleculeDataset:
    """Select molecules with highest predicted activity (exploitation).

    Args:
        dataset: Pool of candidate molecules.
        model: Trained QSARModel.
        batch_size: Number of molecules to select.
        feature_source: Feature type for prediction.
        fp_type: Fingerprint type if using fingerprints.

    Returns:
        MoleculeDataset of selected molecules.
    """
    X, _, indices = _extract_features(dataset, feature_source, fp_type)
    if X.shape[0] == 0:
        raise GenerationError("No valid features for greedy acquisition")

    # Apply scaler if model has one
    X_pred = X.copy()
    if hasattr(model, "scaler") and model.scaler is not None:
        X_pred = model.scaler.transform(X_pred)

    predictions = model.predict(X_pred)

    # Select top-N by predicted value
    n_select = min(batch_size, len(predictions))
    top_indices = np.argsort(predictions)[-n_select:][::-1]

    # Store predictions
    valid = dataset.valid_records
    for local_idx in top_indices:
        real_idx = indices[local_idx]
        if real_idx < len(valid):
            valid[real_idx].properties["al_predicted"] = float(predictions[local_idx])

    selected_valid_indices = [indices[i] for i in top_indices]
    return _select_by_indices(dataset, selected_valid_indices, "greedy")


def uncertainty_acquisition(
    dataset: MoleculeDataset,
    model: Any,
    batch_size: int = AL_DEFAULT_BATCH_SIZE,
    feature_source: str = "descriptors",
    fp_type: Optional[str] = None,
) -> MoleculeDataset:
    """Select molecules with highest prediction uncertainty (exploration).

    Args:
        dataset: Pool of candidate molecules.
        model: Trained QSARModel (tree-based).
        batch_size: Number of molecules to select.
        feature_source: Feature type for prediction.
        fp_type: Fingerprint type.

    Returns:
        MoleculeDataset of selected molecules.
    """
    X, _, indices = _extract_features(dataset, feature_source, fp_type)
    if X.shape[0] == 0:
        raise GenerationError("No valid features for uncertainty acquisition")

    uncertainties = compute_ensemble_uncertainty(model, X)

    n_select = min(batch_size, len(uncertainties))
    top_indices = np.argsort(uncertainties)[-n_select:][::-1]

    valid = dataset.valid_records
    for local_idx in top_indices:
        real_idx = indices[local_idx]
        if real_idx < len(valid):
            valid[real_idx].properties["al_uncertainty"] = float(uncertainties[local_idx])

    selected_valid_indices = [indices[i] for i in top_indices]
    return _select_by_indices(dataset, selected_valid_indices, "uncertainty")


def ucb_acquisition(
    dataset: MoleculeDataset,
    model: Any,
    batch_size: int = AL_DEFAULT_BATCH_SIZE,
    kappa: float = AL_DEFAULT_KAPPA,
    feature_source: str = "descriptors",
    fp_type: Optional[str] = None,
) -> MoleculeDataset:
    """Upper Confidence Bound acquisition: predicted + kappa * uncertainty.

    Balances exploitation (high prediction) with exploration (high uncertainty).

    Args:
        dataset: Pool of candidate molecules.
        model: Trained QSARModel.
        batch_size: Number of molecules to select.
        kappa: Exploration weight (higher = more exploration).
        feature_source: Feature type for prediction.
        fp_type: Fingerprint type.

    Returns:
        MoleculeDataset of selected molecules.
    """
    X, _, indices = _extract_features(dataset, feature_source, fp_type)
    if X.shape[0] == 0:
        raise GenerationError("No valid features for UCB acquisition")

    # Get predictions
    X_pred = X.copy()
    if hasattr(model, "scaler") and model.scaler is not None:
        X_pred = model.scaler.transform(X_pred)
    predictions = model.predict(X_pred)

    # Get uncertainties
    uncertainties = compute_ensemble_uncertainty(model, X)

    # UCB score
    ucb_scores = predictions + kappa * uncertainties

    n_select = min(batch_size, len(ucb_scores))
    top_indices = np.argsort(ucb_scores)[-n_select:][::-1]

    valid = dataset.valid_records
    for local_idx in top_indices:
        real_idx = indices[local_idx]
        if real_idx < len(valid):
            valid[real_idx].properties["al_ucb_score"] = float(ucb_scores[local_idx])
            valid[real_idx].properties["al_predicted"] = float(predictions[local_idx])
            valid[real_idx].properties["al_uncertainty"] = float(uncertainties[local_idx])

    selected_valid_indices = [indices[i] for i in top_indices]
    return _select_by_indices(dataset, selected_valid_indices, "ucb")


def diversity_acquisition(
    dataset: MoleculeDataset,
    batch_size: int = AL_DEFAULT_BATCH_SIZE,
    fp_type: str = "morgan_r2_2048",
) -> MoleculeDataset:
    """Select structurally diverse molecules using MaxMin picking.

    Args:
        dataset: Pool of candidate molecules.
        batch_size: Number of molecules to select.
        fp_type: Fingerprint type for diversity calculation.

    Returns:
        MoleculeDataset of selected molecules.
    """
    fps = _get_fps_matrix(dataset, fp_type)
    if fps.shape[0] == 0:
        raise GenerationError("No valid fingerprints for diversity acquisition")

    n_select = min(batch_size, fps.shape[0])
    selected_indices = maxmin_diversity_pick(fps, n_select)

    return _select_by_indices(dataset, selected_indices, "diversity")


def balanced_acquisition(
    dataset: MoleculeDataset,
    model: Any,
    batch_size: int = AL_DEFAULT_BATCH_SIZE,
    exploration_weight: float = 0.5,
    feature_source: str = "descriptors",
    fp_type: Optional[str] = None,
) -> MoleculeDataset:
    """Balanced acquisition combining exploitation, exploration, and diversity.

    Splits the batch into three parts based on exploration_weight:
    - Exploitation (greedy): (1 - exploration_weight) of batch
    - Exploration (uncertainty): exploration_weight / 2 of batch
    - Diversity: exploration_weight / 2 of batch

    Args:
        dataset: Pool of candidate molecules.
        model: Trained QSARModel.
        batch_size: Total number of molecules to select.
        exploration_weight: Fraction dedicated to exploration + diversity (0-1).
        feature_source: Feature type for prediction.
        fp_type: Fingerprint type.

    Returns:
        MoleculeDataset of selected molecules.
    """
    n_exploit = max(1, int(batch_size * (1 - exploration_weight)))
    n_explore = max(1, int(batch_size * exploration_weight / 2))
    n_diverse = batch_size - n_exploit - n_explore

    selected_indices = set()

    # 1. Greedy selection
    X, _, indices = _extract_features(dataset, feature_source, fp_type)
    if X.shape[0] == 0:
        raise GenerationError("No valid features for balanced acquisition")

    X_pred = X.copy()
    if hasattr(model, "scaler") and model.scaler is not None:
        X_pred = model.scaler.transform(X_pred)
    predictions = model.predict(X_pred)

    sorted_by_pred = np.argsort(predictions)[::-1]
    for idx in sorted_by_pred:
        if len(selected_indices) >= n_exploit:
            break
        selected_indices.add(int(indices[idx]))

    # 2. Uncertainty selection
    try:
        uncertainties = compute_ensemble_uncertainty(model, X)
        sorted_by_unc = np.argsort(uncertainties)[::-1]
        for idx in sorted_by_unc:
            if len(selected_indices) >= n_exploit + n_explore:
                break
            real_idx = int(indices[idx])
            if real_idx not in selected_indices:
                selected_indices.add(real_idx)
    except Exception:
        # Fall back to more greedy if uncertainty not available
        for idx in sorted_by_pred:
            if len(selected_indices) >= n_exploit + n_explore:
                break
            real_idx = int(indices[idx])
            if real_idx not in selected_indices:
                selected_indices.add(real_idx)

    # 3. Diversity selection
    if n_diverse > 0:
        fp_key = fp_type or "morgan_r2_2048"
        fps = _get_fps_matrix(dataset, fp_key)
        if fps.shape[0] > 0:
            diverse_picks = maxmin_diversity_pick(fps, n_diverse + len(selected_indices))
            for idx in diverse_picks:
                if len(selected_indices) >= batch_size:
                    break
                if idx not in selected_indices:
                    selected_indices.add(idx)

    return _select_by_indices(dataset, list(selected_indices), "balanced")

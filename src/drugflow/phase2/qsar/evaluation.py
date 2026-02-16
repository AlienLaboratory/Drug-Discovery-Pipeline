"""QSAR model evaluation: metrics, cross-validation, Y-randomization, AD.

Provides comprehensive evaluation of trained QSAR models including
cross-validation, Y-randomization test, and applicability domain.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from drugflow.core.constants import QSAR_DEFAULT_CV_FOLDS, QSAR_Y_RANDOMIZATION_ITERATIONS
from drugflow.core.exceptions import ModelError
from drugflow.core.logging import get_logger

logger = get_logger("qsar.evaluation")


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "regression",
) -> Dict[str, float]:
    """Compute evaluation metrics for predictions.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    task : str
        "regression" or "classification".

    Returns
    -------
    dict
        Mapping of metric name to value.
    """
    metrics = {}

    if task == "regression":
        from sklearn.metrics import (
            r2_score,
            mean_squared_error,
            mean_absolute_error,
        )
        metrics["r2"] = float(r2_score(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))

        # Pearson correlation
        if len(y_true) > 1:
            corr = np.corrcoef(y_true, y_pred)[0, 1]
            metrics["pearson_r"] = float(corr) if np.isfinite(corr) else 0.0
        else:
            metrics["pearson_r"] = 0.0

    elif task == "classification":
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            matthews_corrcoef,
        )

        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        n_classes = len(set(y_true))
        avg = "binary" if n_classes == 2 else "weighted"
        metrics["precision"] = float(
            precision_score(y_true, y_pred, average=avg, zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(y_true, y_pred, average=avg, zero_division=0)
        )
        metrics["f1"] = float(
            f1_score(y_true, y_pred, average=avg, zero_division=0)
        )
        if n_classes == 2:
            metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    return metrics


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "random_forest",
    task: str = "regression",
    n_folds: int = QSAR_DEFAULT_CV_FOLDS,
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Perform k-fold cross-validation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    model_type : str
        Model type name.
    task : str
        "regression" or "classification".
    n_folds : int
        Number of CV folds.
    params : dict, optional
        Model hyperparameters.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        Cross-validation results with per-fold and mean metrics.
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    from drugflow.phase2.qsar.models import _get_estimator

    if task == "classification":
        kf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )
    else:
        kf = KFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )

    fold_metrics = []
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        estimator = _get_estimator(model_type, task, params)
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)

        fold_result = evaluate_model(y_test, y_pred, task)
        fold_result["fold"] = fold
        fold_metrics.append(fold_result)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

    # Aggregate metrics
    metric_names = [k for k in fold_metrics[0] if k != "fold"]
    mean_metrics = {}
    std_metrics = {}
    for name in metric_names:
        vals = [fm[name] for fm in fold_metrics]
        mean_metrics[name] = float(np.mean(vals))
        std_metrics[name] = float(np.std(vals))

    # Overall metrics on concatenated predictions
    overall = evaluate_model(
        np.array(all_y_true), np.array(all_y_pred), task
    )

    result = {
        "n_folds": n_folds,
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "overall_metrics": overall,
    }

    primary_metric = "r2" if task == "regression" else "accuracy"
    logger.info(
        f"CV ({n_folds}-fold): mean {primary_metric} = "
        f"{mean_metrics.get(primary_metric, 0):.4f} "
        f"(+/- {std_metrics.get(primary_metric, 0):.4f})"
    )
    return result


def y_randomization_test(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "random_forest",
    task: str = "regression",
    n_iterations: int = QSAR_Y_RANDOMIZATION_ITERATIONS,
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Perform Y-randomization test to validate model is not overfitting.

    Shuffles labels and retrains the model multiple times. A valid model
    should perform significantly better than randomized versions.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    model_type : str
        Model type.
    task : str
        "regression" or "classification".
    n_iterations : int
        Number of randomization iterations.
    params : dict, optional
        Model hyperparameters.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        Y-randomization results.
    """
    from drugflow.phase2.qsar.models import _get_estimator

    rng = np.random.RandomState(random_state)

    # Train original model
    estimator = _get_estimator(model_type, task, params)
    estimator.fit(X, y)
    y_pred_original = estimator.predict(X)
    original_metrics = evaluate_model(y, y_pred_original, task)

    # Randomized iterations
    randomized_metrics = []
    for i in range(n_iterations):
        y_shuffled = rng.permutation(y)
        estimator = _get_estimator(model_type, task, params)
        estimator.fit(X, y_shuffled)
        y_pred_rand = estimator.predict(X)
        rand_metrics = evaluate_model(y_shuffled, y_pred_rand, task)
        randomized_metrics.append(rand_metrics)

    # Summarize
    primary_metric = "r2" if task == "regression" else "accuracy"
    original_score = original_metrics.get(primary_metric, 0)
    rand_scores = [m.get(primary_metric, 0) for m in randomized_metrics]
    mean_rand = float(np.mean(rand_scores))
    std_rand = float(np.std(rand_scores))

    # Is the model valid? Original should be significantly better
    is_valid = original_score > mean_rand + 2 * std_rand

    result = {
        "original_metrics": original_metrics,
        "mean_randomized_metrics": {
            name: float(np.mean([m.get(name, 0) for m in randomized_metrics]))
            for name in original_metrics
        },
        "std_randomized_metrics": {
            name: float(np.std([m.get(name, 0) for m in randomized_metrics]))
            for name in original_metrics
        },
        "n_iterations": n_iterations,
        "is_valid": is_valid,
        "primary_metric": primary_metric,
        "original_score": original_score,
        "mean_randomized_score": mean_rand,
    }

    status = "PASSED" if is_valid else "FAILED"
    logger.info(
        f"Y-randomization ({status}): original {primary_metric}={original_score:.4f}, "
        f"randomized mean={mean_rand:.4f} (+/- {std_rand:.4f})"
    )
    return result


def compute_applicability_domain(
    X_train: np.ndarray,
    X_query: np.ndarray,
    method: str = "leverage",
    threshold_factor: float = 3.0,
) -> Dict[str, Any]:
    """Compute applicability domain for query molecules.

    Determines whether query molecules are within the chemical space
    of the training set.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    X_query : np.ndarray
        Query feature matrix.
    method : str
        AD method: "leverage" (Williams plot), "distance" (Euclidean).
    threshold_factor : float
        Factor for threshold computation.

    Returns
    -------
    dict
        AD results with per-molecule scores and in/out flags.
    """
    if method == "leverage":
        return _leverage_ad(X_train, X_query, threshold_factor)
    elif method == "distance":
        return _distance_ad(X_train, X_query, threshold_factor)
    else:
        raise ModelError(f"Unknown AD method: '{method}'. Use 'leverage' or 'distance'.")


def _leverage_ad(
    X_train: np.ndarray,
    X_query: np.ndarray,
    threshold_factor: float = 3.0,
) -> Dict[str, Any]:
    """Leverage-based applicability domain (Williams plot)."""
    n, p = X_train.shape

    # Hat matrix: H = X(X^TX)^{-1}X^T
    try:
        XtX_inv = np.linalg.pinv(X_train.T @ X_train)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse
        XtX_inv = np.linalg.pinv(X_train.T @ X_train)

    # Threshold: h* = 3p/n
    h_threshold = threshold_factor * p / n

    # Compute leverage for query molecules
    leverages = np.array([
        float(x @ XtX_inv @ x.T) for x in X_query
    ])

    in_domain = leverages <= h_threshold

    return {
        "method": "leverage",
        "leverages": leverages.tolist(),
        "threshold": float(h_threshold),
        "in_domain": in_domain.tolist(),
        "n_in_domain": int(in_domain.sum()),
        "n_out_domain": int((~in_domain).sum()),
        "fraction_in_domain": float(in_domain.mean()),
    }


def _distance_ad(
    X_train: np.ndarray,
    X_query: np.ndarray,
    threshold_factor: float = 3.0,
) -> Dict[str, Any]:
    """Euclidean distance-based applicability domain."""
    # Compute centroid and mean distance in training set
    centroid = X_train.mean(axis=0)
    train_distances = np.linalg.norm(X_train - centroid, axis=1)
    mean_dist = float(train_distances.mean())
    std_dist = float(train_distances.std())

    # Threshold: mean + factor * std
    threshold = mean_dist + threshold_factor * std_dist

    # Compute distances for query molecules
    query_distances = np.linalg.norm(X_query - centroid, axis=1)
    in_domain = query_distances <= threshold

    return {
        "method": "distance",
        "distances": query_distances.tolist(),
        "threshold": threshold,
        "in_domain": in_domain.tolist(),
        "n_in_domain": int(in_domain.sum()),
        "n_out_domain": int((~in_domain).sum()),
        "fraction_in_domain": float(in_domain.mean()),
    }

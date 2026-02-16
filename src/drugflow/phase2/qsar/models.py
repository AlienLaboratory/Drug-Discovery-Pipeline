"""QSAR model training and management.

Provides a QSARModel wrapper around scikit-learn estimators with
metadata tracking, and a train_model() function that builds
and fits models for regression or classification tasks.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from drugflow.core.constants import QSAR_DEFAULT_PARAMS, QSAR_MODEL_TYPES, QSAR_TASK_TYPES
from drugflow.core.exceptions import ModelError
from drugflow.core.logging import get_logger

logger = get_logger("qsar.models")


def _check_xgboost_available() -> bool:
    """Check if XGBoost is installed."""
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


# Estimator registry
def _get_estimator(
    model_type: str,
    task: str,
    params: Optional[Dict[str, Any]] = None,
):
    """Create a scikit-learn estimator based on model type and task.

    Parameters
    ----------
    model_type : str
        Model type: "random_forest", "gradient_boosting", or "xgboost".
    task : str
        Task type: "regression" or "classification".
    params : dict, optional
        Custom hyperparameters (override defaults).

    Returns
    -------
    estimator
        Fitted scikit-learn compatible estimator.
    """
    # Merge with defaults
    default_params = QSAR_DEFAULT_PARAMS.get(model_type, {}).copy()
    if params:
        default_params.update(params)

    if model_type == "random_forest":
        if task == "regression":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**default_params)
        else:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**default_params)

    elif model_type == "gradient_boosting":
        if task == "regression":
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(**default_params)
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(**default_params)

    elif model_type == "xgboost":
        if not _check_xgboost_available():
            raise ModelError(
                "XGBoost is not installed. Install with: pip install xgboost"
            )
        import xgboost as xgb
        if task == "regression":
            return xgb.XGBRegressor(**default_params)
        else:
            return xgb.XGBClassifier(**default_params)

    else:
        raise ModelError(
            f"Unknown model type: '{model_type}'. "
            f"Available: {QSAR_MODEL_TYPES}"
        )


@dataclass
class QSARModel:
    """Wrapper around a trained QSAR model with metadata.

    Attributes
    ----------
    estimator : object
        Fitted scikit-learn estimator.
    model_type : str
        Model type name (e.g. "random_forest").
    task : str
        Task type ("regression" or "classification").
    feature_names : list of str
        Feature names used during training.
    scaler : object, optional
        Fitted StandardScaler (None if features not scaled).
    metadata : dict
        Additional metadata (training date, dataset info, etc.).
    training_metrics : dict
        Training performance metrics.
    """
    estimator: Any
    model_type: str
    task: str
    feature_names: List[str]
    scaler: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    training_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def is_classifier(self) -> bool:
        return self.task == "classification"

    @property
    def is_regressor(self) -> bool:
        return self.task == "regression"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predictions.
        """
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Class probabilities.

        Raises
        ------
        ModelError
            If model is not a classifier.
        """
        if not self.is_classifier:
            raise ModelError("predict_proba only available for classifiers")
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.estimator.predict_proba(X)

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores.

        Returns
        -------
        dict or None
            Mapping of feature name to importance score.
        """
        if hasattr(self.estimator, "feature_importances_"):
            importances = self.estimator.feature_importances_
            return dict(zip(self.feature_names, importances))
        return None

    def summary(self) -> Dict[str, Any]:
        """Get model summary information."""
        info = {
            "model_type": self.model_type,
            "task": self.task,
            "n_features": self.n_features,
            "training_metrics": self.training_metrics,
            "metadata": self.metadata,
        }
        importances = self.get_feature_importances()
        if importances:
            sorted_imp = sorted(
                importances.items(), key=lambda x: x[1], reverse=True
            )
            info["top_features"] = sorted_imp[:10]
        return info


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "random_forest",
    task: str = "regression",
    params: Optional[Dict[str, Any]] = None,
    feature_names: Optional[List[str]] = None,
    scaler: Any = None,
    dataset_name: str = "",
) -> QSARModel:
    """Train a QSAR model.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    model_type : str
        Model type: "random_forest", "gradient_boosting", "xgboost".
    task : str
        "regression" or "classification".
    params : dict, optional
        Custom hyperparameters.
    feature_names : list of str, optional
        Feature names.
    scaler : object, optional
        Fitted StandardScaler.
    dataset_name : str
        Name of training dataset for metadata.

    Returns
    -------
    QSARModel
        Trained model wrapper.

    Raises
    ------
    ModelError
        If training fails.
    """
    if task not in QSAR_TASK_TYPES:
        raise ModelError(f"Unknown task: '{task}'. Use: {QSAR_TASK_TYPES}")
    if model_type not in QSAR_MODEL_TYPES:
        raise ModelError(
            f"Unknown model_type: '{model_type}'. Use: {QSAR_MODEL_TYPES}"
        )

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    logger.info(
        f"Training {model_type} ({task}) with "
        f"{X_train.shape[0]} samples, {X_train.shape[1]} features"
    )

    try:
        estimator = _get_estimator(model_type, task, params)
        estimator.fit(X_train, y_train)
    except Exception as e:
        raise ModelError(f"Model training failed: {e}")

    # Compute training metrics
    y_pred = estimator.predict(X_train)
    training_metrics = _compute_quick_metrics(y_train, y_pred, task)

    model = QSARModel(
        estimator=estimator,
        model_type=model_type,
        task=task,
        feature_names=feature_names,
        scaler=scaler,
        metadata={
            "trained_at": datetime.now().isoformat(),
            "n_train_samples": int(X_train.shape[0]),
            "n_features": int(X_train.shape[1]),
            "dataset_name": dataset_name,
            "params": params or {},
        },
        training_metrics=training_metrics,
    )

    logger.info(f"Training complete. Metrics: {training_metrics}")
    return model


def _compute_quick_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
) -> Dict[str, float]:
    """Compute quick training metrics."""
    metrics = {}
    if task == "regression":
        from sklearn.metrics import r2_score, mean_squared_error
        metrics["r2"] = float(r2_score(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    else:
        from sklearn.metrics import accuracy_score
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    return metrics

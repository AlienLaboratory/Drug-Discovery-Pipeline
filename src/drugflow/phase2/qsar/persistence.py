"""QSAR model persistence: save and load trained models.

Uses joblib for efficient serialization of scikit-learn models.
"""

from pathlib import Path
from typing import Optional

import joblib

from drugflow.core.exceptions import ModelError
from drugflow.core.logging import get_logger
from drugflow.phase2.qsar.models import QSARModel

logger = get_logger("qsar.persistence")


def save_model(model: QSARModel, path: str) -> str:
    """Save a trained QSARModel to disk.

    Parameters
    ----------
    model : QSARModel
        Trained model to save.
    path : str
        Output file path (typically .joblib extension).

    Returns
    -------
    str
        Absolute path of saved file.

    Raises
    ------
    ModelError
        If save fails.
    """
    try:
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save as dict for portability
        model_data = {
            "estimator": model.estimator,
            "model_type": model.model_type,
            "task": model.task,
            "feature_names": model.feature_names,
            "scaler": model.scaler,
            "metadata": model.metadata,
            "training_metrics": model.training_metrics,
            "version": "1.0",
        }

        joblib.dump(model_data, str(filepath))
        logger.info(f"Model saved to {filepath}")
        return str(filepath.resolve())

    except Exception as e:
        raise ModelError(f"Failed to save model to {path}: {e}")


def load_model(path: str) -> QSARModel:
    """Load a trained QSARModel from disk.

    Parameters
    ----------
    path : str
        Path to saved model file.

    Returns
    -------
    QSARModel
        Loaded model.

    Raises
    ------
    ModelError
        If load fails or file not found.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise ModelError(f"Model file not found: {path}")

    try:
        model_data = joblib.load(str(filepath))
    except Exception as e:
        raise ModelError(f"Failed to load model from {path}: {e}")

    # Validate required keys
    required_keys = {"estimator", "model_type", "task", "feature_names"}
    missing = required_keys - set(model_data.keys())
    if missing:
        raise ModelError(
            f"Invalid model file, missing keys: {missing}"
        )

    model = QSARModel(
        estimator=model_data["estimator"],
        model_type=model_data["model_type"],
        task=model_data["task"],
        feature_names=model_data["feature_names"],
        scaler=model_data.get("scaler"),
        metadata=model_data.get("metadata", {}),
        training_metrics=model_data.get("training_metrics", {}),
    )

    logger.info(
        f"Model loaded from {filepath}: "
        f"{model.model_type} ({model.task}), "
        f"{model.n_features} features"
    )
    return model

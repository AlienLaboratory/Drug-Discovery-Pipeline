"""Tests for SVR support and model comparison functionality."""

import csv
import os
import tempfile

import numpy as np
import pytest

from drugflow.core.exceptions import ModelError
from drugflow.phase2.qsar.models import train_model, _get_estimator


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def regression_data():
    """Simple regression dataset."""
    np.random.seed(42)
    X = np.random.rand(80, 5)
    y = X[:, 0] * 2 + X[:, 1] + np.random.rand(80) * 0.1
    return X, y


@pytest.fixture
def classification_data():
    """Simple binary classification dataset."""
    np.random.seed(42)
    X = np.random.rand(80, 5)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(float)
    return X, y


@pytest.fixture
def tmp_dir():
    """Temporary directory for output."""
    with tempfile.TemporaryDirectory() as d:
        yield d


# ── TestSVRSupport ──────────────────────────────────────────


class TestSVRSupport:
    """Tests for SVR/SVC model type."""

    def test_svr_regression_estimator(self):
        """_get_estimator returns SVR for regression."""
        from sklearn.svm import SVR
        est = _get_estimator("svr", "regression")
        assert isinstance(est, SVR)

    def test_svc_classification_estimator(self):
        """_get_estimator returns SVC for classification."""
        from sklearn.svm import SVC
        est = _get_estimator("svr", "classification")
        assert isinstance(est, SVC)
        # Must have probability=True for predict_proba
        assert est.probability is True

    def test_train_svr_regression(self, regression_data):
        """Train SVR regressor via train_model."""
        X, y = regression_data
        model = train_model(X, y, model_type="svr", task="regression")
        assert model.model_type == "svr"
        assert model.task == "regression"
        assert model.n_features == 5

    def test_train_svr_classification(self, classification_data):
        """Train SVC classifier via train_model."""
        X, y = classification_data
        model = train_model(X, y, model_type="svr", task="classification")
        assert model.model_type == "svr"
        assert model.task == "classification"

    def test_svr_predict(self, regression_data):
        """SVR model can predict."""
        X, y = regression_data
        model = train_model(X, y, model_type="svr", task="regression")
        preds = model.predict(X[:5])
        assert preds.shape == (5,)

    def test_svc_predict_proba(self, classification_data):
        """SVC model can predict probabilities."""
        X, y = classification_data
        model = train_model(X, y, model_type="svr", task="classification")
        proba = model.predict_proba(X[:5])
        assert proba.shape[0] == 5
        assert proba.shape[1] == 2  # binary

    def test_svr_no_feature_importances(self, regression_data):
        """SVR does not have feature_importances_."""
        X, y = regression_data
        model = train_model(X, y, model_type="svr", task="regression")
        assert model.get_feature_importances() is None

    def test_svr_with_custom_params(self, regression_data):
        """SVR accepts custom hyperparameters."""
        X, y = regression_data
        model = train_model(
            X, y, model_type="svr", task="regression",
            params={"kernel": "linear", "C": 10.0},
        )
        assert model.model_type == "svr"

    def test_svr_in_model_types(self):
        """SVR is listed in QSAR_MODEL_TYPES."""
        from drugflow.core.constants import QSAR_MODEL_TYPES
        assert "svr" in QSAR_MODEL_TYPES

    def test_svr_has_default_params(self):
        """SVR has default params in constants."""
        from drugflow.core.constants import QSAR_DEFAULT_PARAMS
        assert "svr" in QSAR_DEFAULT_PARAMS
        assert "kernel" in QSAR_DEFAULT_PARAMS["svr"]


# ── TestModelCompare ────────────────────────────────────────


class TestModelCompare:
    """Tests for the model comparison functionality."""

    def test_compare_cv_all_models(self, regression_data):
        """Cross-validate multiple model types."""
        from drugflow.phase2.qsar.evaluation import cross_validate
        X, y = regression_data

        results = {}
        for mt in ["random_forest", "gradient_boosting", "svr"]:
            cv = cross_validate(X, y, model_type=mt, task="regression", n_folds=3)
            results[mt] = cv["mean_metrics"]["r2"]

        # All should produce some R2
        for mt, r2 in results.items():
            assert isinstance(r2, float), f"{mt} R2 is not float"

    def test_compare_y_randomization(self, regression_data):
        """Y-randomization works for all model types."""
        from drugflow.phase2.qsar.evaluation import y_randomization_test
        X, y = regression_data

        for mt in ["random_forest", "gradient_boosting", "svr"]:
            result = y_randomization_test(
                X, y, model_type=mt, task="regression",
                n_iterations=3,
            )
            assert "is_valid" in result
            assert "original_score" in result

    def test_compare_best_selection(self, regression_data):
        """Best model can be selected from results."""
        from drugflow.phase2.qsar.evaluation import cross_validate
        X, y = regression_data

        results = []
        for mt in ["random_forest", "gradient_boosting", "svr"]:
            cv = cross_validate(X, y, model_type=mt, task="regression", n_folds=3)
            results.append({
                "model_type": mt,
                "cv_r2": cv["mean_metrics"]["r2"],
            })

        best = max(results, key=lambda r: r["cv_r2"])
        assert best["model_type"] in ["random_forest", "gradient_boosting", "svr"]

    def test_compare_classification(self, classification_data):
        """Compare works for classification task."""
        from drugflow.phase2.qsar.evaluation import cross_validate
        X, y = classification_data

        results = {}
        for mt in ["random_forest", "gradient_boosting", "svr"]:
            cv = cross_validate(X, y, model_type=mt, task="classification", n_folds=3)
            results[mt] = cv["mean_metrics"]["accuracy"]

        for mt, acc in results.items():
            assert 0 <= acc <= 1, f"{mt} accuracy out of range"


# ── TestCLICompare ──────────────────────────────────────────


class TestCLICompare:
    """Tests for the `model compare` CLI command."""

    def _make_csv(self, tmp_dir, n=60):
        """Create a simple CSV with SMILES and activity."""
        from rdkit import Chem
        smiles_list = [
            "c1ccccc1", "CCO", "CC(=O)O", "c1ccncc1", "CC(C)O",
            "CCCC", "c1ccc(O)cc1", "CC(=O)N", "CCC(=O)O", "CCCO",
            "c1ccc(N)cc1", "CC=O", "CCS", "CCN", "CCCN",
            "c1ccc(F)cc1", "c1ccc(Cl)cc1", "c1ccc(Br)cc1",
            "CC(C)(C)O", "CCOCC",
        ]
        path = os.path.join(tmp_dir, "test_data.csv")
        np.random.seed(42)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["smiles", "activity"])
            for i in range(n):
                smi = smiles_list[i % len(smiles_list)]
                act = np.random.rand() * 3 + 5
                writer.writerow([smi, f"{act:.4f}"])
        return path

    def test_cli_compare_basic(self, tmp_dir):
        """Basic model compare CLI invocation."""
        from click.testing import CliRunner
        from drugflow.cli.model_commands import compare

        csv_path = self._make_csv(tmp_dir)
        out_dir = os.path.join(tmp_dir, "compare_out")

        runner = CliRunner()
        result = runner.invoke(compare, [
            "-i", csv_path,
            "--activity-col", "activity",
            "--cv", "3",
            "--no-y-randomization",
            "-o", out_dir,
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "COMPARISON RESULTS" in result.output
        assert "BEST" in result.output

    def test_cli_compare_saves_files(self, tmp_dir):
        """Compare saves comparison.csv and best_model.joblib."""
        from click.testing import CliRunner
        from drugflow.cli.model_commands import compare

        csv_path = self._make_csv(tmp_dir)
        out_dir = os.path.join(tmp_dir, "compare_out2")

        runner = CliRunner()
        result = runner.invoke(compare, [
            "-i", csv_path,
            "--activity-col", "activity",
            "--cv", "3",
            "--no-y-randomization",
            "-o", out_dir,
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert os.path.exists(os.path.join(out_dir, "comparison.csv"))
        assert os.path.exists(os.path.join(out_dir, "best_model.joblib"))

    def test_cli_compare_with_y_rand(self, tmp_dir):
        """Compare with Y-randomization enabled."""
        from click.testing import CliRunner
        from drugflow.cli.model_commands import compare

        csv_path = self._make_csv(tmp_dir)
        out_dir = os.path.join(tmp_dir, "compare_yrand")

        runner = CliRunner()
        result = runner.invoke(compare, [
            "-i", csv_path,
            "--activity-col", "activity",
            "--cv", "3",
            "--y-randomization",
            "-o", out_dir,
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Y-rand" in result.output

    def test_cli_compare_csv_content(self, tmp_dir):
        """Comparison CSV has all model types."""
        from click.testing import CliRunner
        from drugflow.cli.model_commands import compare

        csv_path = self._make_csv(tmp_dir)
        out_dir = os.path.join(tmp_dir, "compare_csv")

        runner = CliRunner()
        result = runner.invoke(compare, [
            "-i", csv_path,
            "--activity-col", "activity",
            "--cv", "3",
            "--no-y-randomization",
            "-o", out_dir,
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        comp_path = os.path.join(out_dir, "comparison.csv")
        with open(comp_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        model_types = [r["model_type"] for r in rows]
        assert "random_forest" in model_types
        assert "gradient_boosting" in model_types
        assert "svr" in model_types

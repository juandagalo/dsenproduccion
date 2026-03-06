"""Tests for churn model evaluation module."""

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from model.evaluation import (
    build_threshold_tradeoff_table,
    classify_with_threshold,
    compute_threshold_metrics,
    extract_feature_coefficients,
    find_optimal_threshold,
    load_evaluation_config,
    save_evaluation_results,
)
from model.training import build_model_pipelines, train_final_model


@pytest.fixture()
def feature_config() -> dict[str, Any]:
    """Minimal feature config matching primary schema."""
    return {
        "numeric_columns": ["tenure", "MonthlyCharges"],
        "boolean_columns": ["SeniorCitizen", "Partner", "Dependents", "PaperlessBilling"],
        "consolidate_ohe_columns": [
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ],
        "multi_ohe_columns": ["InternetService", "PaymentMethod"],
        "contract_column": "Contract",
        "contract_order": ["Month-to-month", "One year", "Two year"],
        "service_level_replacements": {
            "No internet service": "No",
            "No phone service": "No",
        },
        "new_customer_threshold": 6,
        "numeric_impute_strategy": "median",
        "categorical_impute_strategy": "most_frequent",
        "boolean_impute_strategy": "most_frequent",
    }


@pytest.fixture()
def primary_df() -> pd.DataFrame:
    """Small DataFrame mimicking the primary churn schema (20 rows)."""
    n = 20
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "SeniorCitizen": rng.choice([True, False], n),
            "Partner": rng.choice([True, False], n),
            "Dependents": rng.choice([True, False], n),
            "tenure": pd.array(rng.integers(0, 72, n), dtype="Int16"),
            "MultipleLines": pd.Categorical(rng.choice(["Yes", "No", "No phone service"], n)),
            "InternetService": pd.Categorical(rng.choice(["DSL", "Fiber optic", "No"], n)),
            "OnlineSecurity": pd.Categorical(rng.choice(["Yes", "No", "No internet service"], n)),
            "OnlineBackup": pd.Categorical(rng.choice(["Yes", "No", "No internet service"], n)),
            "DeviceProtection": pd.Categorical(rng.choice(["Yes", "No", "No internet service"], n)),
            "TechSupport": pd.Categorical(rng.choice(["Yes", "No", "No internet service"], n)),
            "StreamingTV": pd.Categorical(rng.choice(["Yes", "No", "No internet service"], n)),
            "StreamingMovies": pd.Categorical(rng.choice(["Yes", "No", "No internet service"], n)),
            "Contract": pd.Categorical(rng.choice(["Month-to-month", "One year", "Two year"], n)),
            "PaperlessBilling": rng.choice([True, False], n),
            "PaymentMethod": pd.Categorical(
                rng.choice(
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    n,
                )
            ),
            "MonthlyCharges": rng.uniform(18.0, 120.0, n).round(2),
            "Churn": rng.choice([0, 1], n),
        }
    )


@pytest.fixture()
def xy_split(primary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split primary_df into X and y."""
    X = primary_df.drop(columns=["Churn"])
    y = primary_df["Churn"]
    return X, y


@pytest.fixture()
def fitted_pipeline(
    feature_config: dict[str, Any],
    xy_split: tuple[pd.DataFrame, pd.Series],
) -> Pipeline:
    """A fitted LR pipeline for testing."""
    train_config: dict[str, Any] = {
        "models": {
            "logistic_regression": {
                "class": "sklearn.linear_model.LogisticRegression",
                "needs_scaler": True,
                "params": {
                    "class_weight": "balanced",
                    "solver": "lbfgs",
                    "max_iter": 1000,
                    "random_state": 42,
                },
            },
        },
    }
    pipelines = build_model_pipelines(feature_config, train_config)
    X, y = xy_split
    return train_final_model(pipelines["logistic_regression"], X, y)


@pytest.fixture()
def y_proba(
    fitted_pipeline: Pipeline,
    xy_split: tuple[pd.DataFrame, pd.Series],
) -> NDArray[np.floating[Any]]:
    """Predicted probabilities for the positive class."""
    X, _ = xy_split
    proba: NDArray[np.floating[Any]] = fitted_pipeline.predict_proba(X)[:, 1]
    return proba


@pytest.fixture()
def threshold_metrics(
    xy_split: tuple[pd.DataFrame, pd.Series],
    y_proba: NDArray[np.floating[Any]],
) -> pd.DataFrame:
    """Threshold metrics DataFrame for testing."""
    _, y = xy_split
    return compute_threshold_metrics(y.values, y_proba)


class TestLoadEvaluationConfig:
    """Tests for load_evaluation_config."""

    def test_loads_default_config(self) -> None:
        config = load_evaluation_config()
        assert "tuned_model_path" in config

    def test_loads_custom_path(self, tmp_path: Any) -> None:
        expected_metric = "precision"
        custom = {
            "tuned_model_path": "models/test.joblib",
            "optimization_metric": expected_metric,
        }
        path = tmp_path / "custom_eval.yml"
        with open(path, "w") as f:
            yaml.safe_dump(custom, f)
        config = load_evaluation_config(path)
        assert config["optimization_metric"] == expected_metric


class TestComputeThresholdMetrics:
    """Tests for compute_threshold_metrics."""

    def test_returns_dataframe(self, threshold_metrics: pd.DataFrame) -> None:
        assert isinstance(threshold_metrics, pd.DataFrame)

    def test_expected_columns(self, threshold_metrics: pd.DataFrame) -> None:
        expected = {"threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn"}
        assert expected.issubset(threshold_metrics.columns)

    def test_row_count(self, threshold_metrics: pd.DataFrame) -> None:
        expected_rows = 98
        assert len(threshold_metrics) == expected_rows

    def test_valid_range(self, threshold_metrics: pd.DataFrame) -> None:
        for col in ("precision", "recall", "f1"):
            assert (threshold_metrics[col] >= 0.0).all()
            assert (threshold_metrics[col] <= 1.0).all()


class TestFindOptimalThreshold:
    """Tests for find_optimal_threshold."""

    def test_returns_tuple(self, threshold_metrics: pd.DataFrame) -> None:
        expected_length = 2
        result = find_optimal_threshold(threshold_metrics)
        assert isinstance(result, tuple)
        assert len(result) == expected_length

    def test_valid_range(self, threshold_metrics: pd.DataFrame) -> None:
        threshold, _f1 = find_optimal_threshold(threshold_metrics)
        assert 0.0 <= threshold <= 1.0

    def test_f1_maximized(self, threshold_metrics: pd.DataFrame) -> None:
        _threshold, best_f1 = find_optimal_threshold(threshold_metrics)
        assert (threshold_metrics["f1"] <= best_f1 + 1e-9).all()


class TestExtractFeatureCoefficients:
    """Tests for extract_feature_coefficients."""

    def test_returns_dataframe(self, fitted_pipeline: Pipeline) -> None:
        result = extract_feature_coefficients(fitted_pipeline)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, fitted_pipeline: Pipeline) -> None:
        result = extract_feature_coefficients(fitted_pipeline)
        expected = {"feature", "coefficient"}
        assert expected.issubset(result.columns)

    def test_sorted_descending(self, fitted_pipeline: Pipeline) -> None:
        result = extract_feature_coefficients(fitted_pipeline)
        abs_coeffs = result["coefficient"].abs().values
        assert (abs_coeffs[:-1] >= abs_coeffs[1:] - 1e-9).all()


class TestBuildThresholdTradeoffTable:
    """Tests for build_threshold_tradeoff_table."""

    def test_returns_dataframe(self, threshold_metrics: pd.DataFrame) -> None:
        result = build_threshold_tradeoff_table(threshold_metrics, [0.3, 0.4, 0.5])
        assert isinstance(result, pd.DataFrame)

    def test_row_count(self, threshold_metrics: pd.DataFrame) -> None:
        candidates = [0.3, 0.4, 0.5]
        result = build_threshold_tradeoff_table(threshold_metrics, candidates)
        assert len(result) == len(candidates)

    def test_expected_columns(self, threshold_metrics: pd.DataFrame) -> None:
        result = build_threshold_tradeoff_table(threshold_metrics, [0.3, 0.4, 0.5])
        expected = {"threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn"}
        assert expected.issubset(result.columns)


class TestClassifyWithThreshold:
    """Tests for classify_with_threshold."""

    def test_binary_output(self, y_proba: NDArray[np.floating[Any]]) -> None:
        result = classify_with_threshold(y_proba, 0.5)
        assert np.isin(result, [0, 1]).all()

    def test_threshold_zero_all_positive(self) -> None:
        probas = np.array([0.1, 0.5, 0.9])
        result = classify_with_threshold(probas, 0.0)
        np.testing.assert_array_equal(result, [1, 1, 1])

    def test_threshold_one_all_negative(self) -> None:
        probas = np.array([0.1, 0.5, 0.9])
        result = classify_with_threshold(probas, 1.0)
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_custom_threshold(self) -> None:
        probas = np.array([0.1, 0.5, 0.9])
        result = classify_with_threshold(probas, 0.5)
        np.testing.assert_array_equal(result, [0, 1, 1])


class TestSaveEvaluationResults:
    """Tests for save_evaluation_results."""

    def test_file_exists(self, tmp_path: Any) -> None:
        results = {"chosen_threshold": 0.45, "f1": 0.72}
        path = save_evaluation_results(results, tmp_path, "eval_results.json")
        assert path.exists()

    def test_has_threshold(self, tmp_path: Any) -> None:
        results = {"chosen_threshold": 0.45, "f1": 0.72}
        path = save_evaluation_results(results, tmp_path, "eval_results.json")
        with open(path) as f:
            data = json.load(f)
        assert "chosen_threshold" in data

    def test_no_numpy_types(self, tmp_path: Any) -> None:
        results: dict[str, Any] = {
            "chosen_threshold": np.float64(0.45),
            "f1": np.float64(0.72),
            "tp": np.int64(10),
            "coefficients": np.array([0.1, 0.2, 0.3]),
        }
        path = save_evaluation_results(results, tmp_path, "eval_results.json")
        with open(path) as f:
            raw = f.read()
        # Re-parse to verify it's valid JSON (no numpy serialization errors)
        data = json.loads(raw)
        assert isinstance(data, dict)

"""Tests for churn inference predict module."""

import json
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline

from inference.predict import (
    build_customer_dataframe,
    get_risk_level,
    load_evaluation_results,
    load_model,
    load_serving_config,
    predict_churn,
)
from model.training import build_model_pipelines, train_final_model


@pytest.fixture()
def serving_config() -> dict[str, Any]:
    """Load default serving configuration."""
    config: dict[str, Any] = load_serving_config()
    return config


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
def sample_customer() -> dict[str, Any]:
    """Single customer feature dictionary with all 16 keys."""
    return {
        "tenure": 24,
        "MonthlyCharges": 65.0,
        "SeniorCitizen": False,
        "Partner": True,
        "Dependents": False,
        "PaperlessBilling": True,
        "MultipleLines": "Yes",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
        "Contract": "Month-to-month",
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
def sample_batch_df(primary_df: pd.DataFrame) -> pd.DataFrame:
    """Primary DataFrame without the Churn target column."""
    return primary_df.drop(columns=["Churn"])


@pytest.fixture()
def fitted_pipeline(
    feature_config: dict[str, Any],
    primary_df: pd.DataFrame,
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
    X = primary_df.drop(columns=["Churn"])
    y = primary_df["Churn"]
    return train_final_model(pipelines["logistic_regression"], X, y)


class TestLoadServingConfig:
    """Tests for load_serving_config."""

    def test_loads_default_config(self, serving_config: dict[str, Any]) -> None:
        assert "model_path" in serving_config

    def test_loads_custom_path(self, tmp_path: Any) -> None:
        custom = {"model_path": "models/custom.joblib", "threshold": 0.5}
        path = tmp_path / "custom_serving.yml"
        with open(path, "w") as f:
            yaml.safe_dump(custom, f)
        config = load_serving_config(path)
        assert config["model_path"] == "models/custom.joblib"


class TestLoadModel:
    """Tests for load_model."""

    def test_returns_pipeline(self, fitted_pipeline: Pipeline, tmp_path: Any) -> None:
        model_path = tmp_path / "test_pipeline.joblib"
        joblib.dump(fitted_pipeline, model_path)
        loaded = load_model(model_path)
        assert isinstance(loaded, Pipeline)

    def test_can_predict(
        self,
        fitted_pipeline: Pipeline,
        sample_batch_df: pd.DataFrame,
        tmp_path: Any,
    ) -> None:
        model_path = tmp_path / "test_pipeline.joblib"
        joblib.dump(fitted_pipeline, model_path)
        loaded = load_model(model_path)
        preds = loaded.predict(sample_batch_df)
        assert preds.shape == (len(sample_batch_df),)


class TestLoadEvaluationResults:
    """Tests for load_evaluation_results."""

    def test_returns_dict(self, tmp_path: Any) -> None:
        data = {"chosen_threshold": 0.59, "f1": 0.72, "roc_auc": 0.848}
        path = tmp_path / "eval_results.json"
        with open(path, "w") as f:
            json.dump(data, f)
        result = load_evaluation_results(path)
        assert isinstance(result, dict)

    def test_has_threshold_key(self, tmp_path: Any) -> None:
        data = {"chosen_threshold": 0.59, "f1": 0.72, "roc_auc": 0.848}
        path = tmp_path / "eval_results.json"
        with open(path, "w") as f:
            json.dump(data, f)
        result = load_evaluation_results(path)
        assert "chosen_threshold" in result


class TestBuildCustomerDataframe:
    """Tests for build_customer_dataframe."""

    def test_returns_dataframe(self, sample_customer: dict[str, Any]) -> None:
        df = build_customer_dataframe(sample_customer)
        assert isinstance(df, pd.DataFrame)

    def test_single_row(self, sample_customer: dict[str, Any]) -> None:
        df = build_customer_dataframe(sample_customer)
        assert len(df) == 1

    def test_correct_columns(self, sample_customer: dict[str, Any]) -> None:
        df = build_customer_dataframe(sample_customer)
        for key in sample_customer:
            assert key in df.columns

    def test_boolean_dtype(self, sample_customer: dict[str, Any]) -> None:
        df = build_customer_dataframe(sample_customer)
        assert df["SeniorCitizen"].dtype == bool


class TestPredictChurn:
    """Tests for predict_churn."""

    def test_returns_dataframe(
        self,
        fitted_pipeline: Pipeline,
        sample_batch_df: pd.DataFrame,
    ) -> None:
        result = predict_churn(fitted_pipeline, sample_batch_df, threshold=0.5)
        assert isinstance(result, pd.DataFrame)

    def test_has_probability_column(
        self,
        fitted_pipeline: Pipeline,
        sample_batch_df: pd.DataFrame,
    ) -> None:
        result = predict_churn(fitted_pipeline, sample_batch_df, threshold=0.5)
        assert "churn_probability" in result.columns

    def test_has_predicted_column(
        self,
        fitted_pipeline: Pipeline,
        sample_batch_df: pd.DataFrame,
    ) -> None:
        result = predict_churn(fitted_pipeline, sample_batch_df, threshold=0.5)
        assert "churn_predicted" in result.columns

    def test_respects_threshold(
        self,
        fitted_pipeline: Pipeline,
        sample_batch_df: pd.DataFrame,
    ) -> None:
        result = predict_churn(fitted_pipeline, sample_batch_df, threshold=0.0)
        assert (result["churn_predicted"] == 1).all()

    def test_probability_range(
        self,
        fitted_pipeline: Pipeline,
        sample_batch_df: pd.DataFrame,
    ) -> None:
        result = predict_churn(fitted_pipeline, sample_batch_df, threshold=0.5)
        assert (result["churn_probability"] >= 0.0).all()
        assert (result["churn_probability"] <= 1.0).all()


class TestGetRiskLevel:
    """Tests for get_risk_level."""

    def test_low(self) -> None:
        assert get_risk_level(0.2) == "Low"

    def test_medium(self) -> None:
        assert get_risk_level(0.45) == "Medium"

    def test_high(self) -> None:
        assert get_risk_level(0.8) == "High"

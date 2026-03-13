"""Tests for inference pipeline entry point."""

import json
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]

from model.training import build_model_pipelines, train_final_model
from pipelines.inference_pipeline.main import main


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
def tmp_inference_env(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    primary_df: pd.DataFrame,
    feature_config: dict[str, Any],
) -> None:
    """Build a full temp directory with all artifacts needed by main()."""
    # Save primary data
    primary_dir = tmp_path / "data" / "03_primary" / "Churn"
    primary_dir.mkdir(parents=True)
    primary_df.to_parquet(primary_dir / "churn_primary.parquet", index=False)

    # Build and save a fitted pipeline
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
    fitted = train_final_model(pipelines["logistic_regression"], X, y)

    model_dir = tmp_path / "models" / "Churn"
    model_dir.mkdir(parents=True)
    joblib.dump(fitted, model_dir / "churn_tuned_pipeline.joblib")

    # Save evaluation results
    eval_results = {"chosen_threshold": 0.59}
    with open(model_dir / "evaluation_results.json", "w") as f:
        json.dump(eval_results, f)

    # Save serving config
    serving_config = {
        "model_path": "models/Churn/churn_tuned_pipeline.joblib",
        "evaluation_results_path": "models/Churn/evaluation_results.json",
        "input_path": "data/03_primary/Churn/churn_primary.parquet",
        "prediction_output_path": "data/07_model_output/Churn",
        "risk_levels": {
            "low_max": 0.3,
            "medium_max": 0.6,
        },
    }
    conf_dir = tmp_path / "conf" / "model_serving"
    conf_dir.mkdir(parents=True)
    with open(conf_dir / "churn.yml", "w") as f:
        yaml.safe_dump(serving_config, f)

    monkeypatch.chdir(tmp_path)


class TestInferencePipelineMain:
    """Tests for inference pipeline main()."""

    def test_main_runs_without_error(self, tmp_inference_env: None) -> None:
        main()

    def test_creates_output_parquet(self, tmp_inference_env: None) -> None:
        main()
        output = Path("data/07_model_output/Churn/predictions.parquet")
        assert output.exists()

    def test_output_has_expected_columns(self, tmp_inference_env: None) -> None:
        main()
        output = pd.read_parquet("data/07_model_output/Churn/predictions.parquet")
        assert "churn_probability" in output.columns
        assert "churn_predicted" in output.columns
        assert "risk_level" in output.columns

    def test_output_row_count_matches_input(self, tmp_inference_env: None) -> None:
        main()
        input_df = pd.read_parquet("data/03_primary/Churn/churn_primary.parquet")
        output = pd.read_parquet("data/07_model_output/Churn/predictions.parquet")
        assert len(output) == len(input_df)

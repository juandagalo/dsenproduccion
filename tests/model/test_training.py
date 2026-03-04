"""Tests for churn model training module."""

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model.training import (
    build_cv_strategy,
    build_model_pipelines,
    compare_cv_results,
    cross_validate_models,
    load_train_config,
    save_cv_results,
    save_model,
    select_best_model,
    train_final_model,
)


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
def train_config() -> dict[str, Any]:
    """Minimal training config for tests."""
    return {
        "cv": {"n_splits": 2, "shuffle": True, "random_state": 42},
        "scoring": ["roc_auc", "f1", "accuracy"],
        "primary_metric": "test_roc_auc",
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
            "random_forest": {
                "class": "sklearn.ensemble.RandomForestClassifier",
                "needs_scaler": False,
                "params": {"class_weight": "balanced", "n_estimators": 10, "random_state": 42},
            },
            "gradient_boosting": {
                "class": "sklearn.ensemble.GradientBoostingClassifier",
                "needs_scaler": False,
                "params": {"n_estimators": 10, "random_state": 42},
            },
        },
        "model_output_path": "data/06_models/Churn",
        "reporting_output_path": "data/08_reporting/Churn",
        "best_pipeline_filename": "churn_best_pipeline.joblib",
        "cv_results_filename": "cv_results.json",
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
def pipelines(feature_config: dict[str, Any], train_config: dict[str, Any]) -> dict[str, Pipeline]:
    """Built model pipelines."""
    result: dict[str, Pipeline] = build_model_pipelines(feature_config, train_config)
    return result


@pytest.fixture()
def cv_results(
    pipelines: dict[str, Pipeline],
    xy_split: tuple[pd.DataFrame, pd.Series],
    train_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Cross-validation results for all models."""
    X, y = xy_split
    result: dict[str, dict[str, Any]] = cross_validate_models(pipelines, X, y, train_config)
    return result


class TestLoadTrainConfig:
    """Tests for load_train_config."""

    def test_loads_default_config(self) -> None:
        config = load_train_config()
        assert "models" in config
        assert "cv" in config
        assert "scoring" in config

    def test_loads_custom_path(self, tmp_path: Any) -> None:
        expected_splits = 3
        custom = {"models": {}, "cv": {"n_splits": expected_splits}}
        path = tmp_path / "custom.yml"
        with open(path, "w") as f:
            yaml.safe_dump(custom, f)
        config = load_train_config(path)
        assert config["cv"]["n_splits"] == expected_splits


class TestBuildModelPipelines:
    """Tests for build_model_pipelines."""

    def test_returns_dict_of_pipelines(self, pipelines: dict[str, Pipeline]) -> None:
        assert isinstance(pipelines, dict)
        for pipeline in pipelines.values():
            assert isinstance(pipeline, Pipeline)

    def test_lr_has_scaler(self, pipelines: dict[str, Pipeline]) -> None:
        lr_pipeline = pipelines["logistic_regression"]
        step_names = [name for name, _ in lr_pipeline.steps]
        assert "scaler" in step_names
        assert isinstance(lr_pipeline.named_steps["scaler"], StandardScaler)

    def test_trees_no_scaler(self, pipelines: dict[str, Pipeline]) -> None:
        for name in ("random_forest", "gradient_boosting"):
            step_names = [s for s, _ in pipelines[name].steps]
            assert "scaler" not in step_names

    def test_each_has_preprocessor(self, pipelines: dict[str, Pipeline]) -> None:
        for pipeline in pipelines.values():
            assert "preprocessor" in pipeline.named_steps

    def test_pipelines_are_independent(self, pipelines: dict[str, Pipeline]) -> None:
        preprocessors = [p.named_steps["preprocessor"] for p in pipelines.values()]
        for i in range(len(preprocessors)):
            for j in range(i + 1, len(preprocessors)):
                assert preprocessors[i] is not preprocessors[j]


class TestBuildCvStrategy:
    """Tests for build_cv_strategy."""

    def test_returns_stratified_kfold(self) -> None:
        cv = build_cv_strategy({"n_splits": 5, "shuffle": True, "random_state": 42})
        assert isinstance(cv, StratifiedKFold)

    def test_params_match_config(self) -> None:
        expected_splits = 3
        expected_seed = 7
        cv = build_cv_strategy(
            {"n_splits": expected_splits, "shuffle": True, "random_state": expected_seed}
        )
        assert cv.n_splits == expected_splits
        assert cv.shuffle is True
        assert cv.random_state == expected_seed


class TestCrossValidateModels:
    """Tests for cross_validate_models."""

    def test_returns_results_for_all_models(self, cv_results: dict[str, dict[str, Any]]) -> None:
        assert set(cv_results.keys()) == {
            "logistic_regression",
            "random_forest",
            "gradient_boosting",
        }

    def test_has_expected_keys(self, cv_results: dict[str, dict[str, Any]]) -> None:
        for result in cv_results.values():
            assert "test_roc_auc" in result
            assert "train_roc_auc" in result
            assert "test_f1" in result

    def test_correct_array_lengths(
        self, cv_results: dict[str, dict[str, Any]], train_config: dict[str, Any]
    ) -> None:
        n_splits = train_config["cv"]["n_splits"]
        for result in cv_results.values():
            assert len(result["test_roc_auc"]) == n_splits

    def test_scores_in_valid_range(self, cv_results: dict[str, dict[str, Any]]) -> None:
        for result in cv_results.values():
            for key in ("test_roc_auc", "test_f1", "test_accuracy"):
                values = result[key]
                assert all(0.0 <= v <= 1.0 for v in values)


class TestCompareCvResults:
    """Tests for compare_cv_results."""

    def test_returns_dataframe(self, cv_results: dict[str, dict[str, Any]]) -> None:
        df = compare_cv_results(cv_results, ["roc_auc", "f1", "accuracy"])
        assert isinstance(df, pd.DataFrame)

    def test_rows_are_models(self, cv_results: dict[str, dict[str, Any]]) -> None:
        df = compare_cv_results(cv_results, ["roc_auc", "f1", "accuracy"])
        assert set(df.index) == {"logistic_regression", "random_forest", "gradient_boosting"}

    def test_columns_contain_metrics(self, cv_results: dict[str, dict[str, Any]]) -> None:
        df = compare_cv_results(cv_results, ["roc_auc", "f1", "accuracy"])
        assert "test_roc_auc_mean" in df.columns
        assert "test_roc_auc_std" in df.columns
        assert "train_roc_auc_mean" in df.columns


class TestSelectBestModel:
    """Tests for select_best_model."""

    def test_returns_string(self, cv_results: dict[str, dict[str, Any]]) -> None:
        best = select_best_model(cv_results, "test_roc_auc")
        assert isinstance(best, str)
        assert best in cv_results

    def test_selects_highest_mean(self) -> None:
        mock_results: dict[str, dict[str, Any]] = {
            "model_a": {"test_roc_auc": np.array([0.5, 0.6])},
            "model_b": {"test_roc_auc": np.array([0.8, 0.9])},
        }
        assert select_best_model(mock_results, "test_roc_auc") == "model_b"


class TestTrainFinalModel:
    """Tests for train_final_model."""

    def test_returns_fitted_pipeline(
        self,
        pipelines: dict[str, Pipeline],
        xy_split: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = xy_split
        pipeline = pipelines["logistic_regression"]
        fitted = train_final_model(pipeline, X, y)
        assert isinstance(fitted, Pipeline)

    def test_predict_shape(
        self,
        pipelines: dict[str, Pipeline],
        xy_split: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = xy_split
        pipeline = pipelines["random_forest"]
        fitted = train_final_model(pipeline, X, y)
        preds = fitted.predict(X.head())
        assert preds.shape == (5,)

    def test_predict_proba_shape(
        self,
        pipelines: dict[str, Pipeline],
        xy_split: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = xy_split
        pipeline = pipelines["gradient_boosting"]
        fitted = train_final_model(pipeline, X, y, needs_sample_weight=True)
        proba = fitted.predict_proba(X.head())
        assert proba.shape == (5, 2)


class TestSaveModel:
    """Tests for save_model."""

    def test_file_exists(
        self,
        pipelines: dict[str, Pipeline],
        xy_split: tuple[pd.DataFrame, pd.Series],
        tmp_path: Any,
    ) -> None:
        X, y = xy_split
        pipeline = train_final_model(pipelines["logistic_regression"], X, y)
        path = save_model(pipeline, tmp_path, "test_model.joblib")
        assert path.exists()

    def test_loaded_model_can_predict(
        self,
        pipelines: dict[str, Pipeline],
        xy_split: tuple[pd.DataFrame, pd.Series],
        tmp_path: Any,
    ) -> None:
        X, y = xy_split
        pipeline = train_final_model(pipelines["logistic_regression"], X, y)
        path = save_model(pipeline, tmp_path, "test_model.joblib")
        loaded = joblib.load(path)
        preds = loaded.predict(X.head())
        assert preds.shape == (5,)


class TestSaveCvResults:
    """Tests for save_cv_results."""

    def test_file_exists(self, cv_results: dict[str, dict[str, Any]], tmp_path: Any) -> None:
        path = save_cv_results(cv_results, "logistic_regression", tmp_path, "results.json")
        assert path.exists()

    def test_json_has_best_model_name(
        self, cv_results: dict[str, dict[str, Any]], tmp_path: Any
    ) -> None:
        path = save_cv_results(cv_results, "logistic_regression", tmp_path, "results.json")
        with open(path) as f:
            data = json.load(f)
        assert data["best_model_name"] == "logistic_regression"

    def test_no_numpy_types(self, cv_results: dict[str, dict[str, Any]], tmp_path: Any) -> None:
        path = save_cv_results(cv_results, "logistic_regression", tmp_path, "results.json")
        with open(path) as f:
            raw = f.read()
        # Re-parse to verify it's valid JSON (no numpy serialization errors)
        data = json.loads(raw)
        for model_data in data["models"].values():
            for value in model_data.values():
                assert not isinstance(value, np.ndarray)

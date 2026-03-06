"""Tests for churn model tuning module."""

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from model.training import build_model_pipelines
from model.tuning import (
    build_param_grid,
    compare_baseline_vs_tuned,
    load_tuning_config,
    run_grid_search,
    save_tuning_results,
    summarize_grid_search,
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
    """Minimal training config for building the LR pipeline."""
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
        },
        "model_output_path": "data/06_models/Churn",
        "reporting_output_path": "data/08_reporting/Churn",
        "best_pipeline_filename": "churn_best_pipeline.joblib",
        "cv_results_filename": "cv_results.json",
    }


@pytest.fixture()
def tuning_config() -> dict[str, Any]:
    """Minimal tuning config with a small parameter grid."""
    return {
        "cv": {"n_splits": 2, "shuffle": True, "random_state": 42},
        "scoring": "roc_auc",
        "refit": True,
        "param_grid": [
            {
                "model__C": [0.1, 1.0],
                "model__l1_ratio": [0.0, 1.0],
                "model__solver": ["saga"],
            },
        ],
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
def lr_pipeline(feature_config: dict[str, Any], train_config: dict[str, Any]) -> Pipeline:
    """Built logistic regression pipeline."""
    pipelines: dict[str, Pipeline] = build_model_pipelines(feature_config, train_config)
    return pipelines["logistic_regression"]


@pytest.fixture()
def grid_search_result(
    lr_pipeline: Pipeline,
    xy_split: tuple[pd.DataFrame, pd.Series],
    tuning_config: dict[str, Any],
) -> GridSearchCV:
    """Fitted GridSearchCV from run_grid_search with small grid."""
    X, y = xy_split
    param_grid = build_param_grid(tuning_config)
    result: GridSearchCV = run_grid_search(lr_pipeline, X, y, param_grid, tuning_config)
    return result


class TestLoadTuningConfig:
    """Tests for load_tuning_config."""

    def test_loads_default_config(self) -> None:
        config = load_tuning_config()
        assert "param_grid" in config
        assert "cv" in config
        assert "scoring" in config

    def test_loads_custom_path(self, tmp_path: Any) -> None:
        expected_splits = 3
        custom = {
            "cv": {"n_splits": expected_splits},
            "scoring": "roc_auc",
            "param_grid": [{"model__C": [1.0]}],
        }
        path = tmp_path / "custom_tuning.yml"
        with open(path, "w") as f:
            yaml.safe_dump(custom, f)
        config = load_tuning_config(path)
        assert config["cv"]["n_splits"] == expected_splits


class TestBuildParamGrid:
    """Tests for build_param_grid."""

    def test_returns_list_of_dicts(self, tuning_config: dict[str, Any]) -> None:
        grid = build_param_grid(tuning_config)
        assert isinstance(grid, list)
        for entry in grid:
            assert isinstance(entry, dict)

    def test_grid_has_expected_keys(self, tuning_config: dict[str, Any]) -> None:
        grid = build_param_grid(tuning_config)
        expected_keys = {"model__l1_ratio", "model__C", "model__solver"}
        for entry in grid:
            assert expected_keys.issubset(entry.keys())


class TestRunGridSearch:
    """Tests for run_grid_search."""

    def test_returns_grid_search_cv(self, grid_search_result: GridSearchCV) -> None:
        assert isinstance(grid_search_result, GridSearchCV)

    def test_best_score_in_valid_range(self, grid_search_result: GridSearchCV) -> None:
        min_score = 0.0
        max_score = 1.0
        assert min_score <= grid_search_result.best_score_ <= max_score

    def test_best_params_has_expected_keys(self, grid_search_result: GridSearchCV) -> None:
        expected_keys = {"model__C", "model__l1_ratio", "model__solver"}
        assert expected_keys.issubset(grid_search_result.best_params_.keys())

    def test_refit_produces_fitted_pipeline(
        self,
        grid_search_result: GridSearchCV,
        xy_split: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, _y = xy_split
        preds = grid_search_result.best_estimator_.predict(X)
        assert preds.shape[0] == len(X)


class TestSummarizeGridSearch:
    """Tests for summarize_grid_search."""

    def test_returns_dataframe(self, grid_search_result: GridSearchCV) -> None:
        df = summarize_grid_search(grid_search_result)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self, grid_search_result: GridSearchCV) -> None:
        df = summarize_grid_search(grid_search_result)
        expected_cols = {"mean_test_score", "std_test_score", "rank_test_score", "params"}
        assert expected_cols.issubset(df.columns)

    def test_sorted_by_rank(self, grid_search_result: GridSearchCV) -> None:
        df = summarize_grid_search(grid_search_result)
        first_rank = 1
        assert df.iloc[0]["rank_test_score"] == first_rank


class TestCompareBaselineVsTuned:
    """Tests for compare_baseline_vs_tuned."""

    def test_returns_dataframe(self, grid_search_result: GridSearchCV) -> None:
        baseline_cv_results: dict[str, dict[str, Any]] = {
            "logistic_regression": {
                "test_roc_auc": np.array([0.6, 0.7]),
            },
        }
        scoring = "roc_auc"
        df = compare_baseline_vs_tuned(
            baseline_cv_results, "logistic_regression", grid_search_result, scoring
        )
        assert isinstance(df, pd.DataFrame)

    def test_has_baseline_and_tuned_rows(self, grid_search_result: GridSearchCV) -> None:
        baseline_cv_results: dict[str, dict[str, Any]] = {
            "logistic_regression": {
                "test_roc_auc": np.array([0.6, 0.7]),
            },
        }
        scoring = "roc_auc"
        df = compare_baseline_vs_tuned(
            baseline_cv_results, "logistic_regression", grid_search_result, scoring
        )
        index_values = set(df.index)
        assert "baseline" in index_values or "logistic_regression" in index_values
        assert "tuned" in index_values or "tuned_best" in index_values

    def test_scores_in_valid_range(self, grid_search_result: GridSearchCV) -> None:
        baseline_cv_results: dict[str, dict[str, Any]] = {
            "logistic_regression": {
                "test_roc_auc": np.array([0.6, 0.7]),
            },
        }
        scoring = "roc_auc"
        df = compare_baseline_vs_tuned(
            baseline_cv_results, "logistic_regression", grid_search_result, scoring
        )
        min_score = 0.0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert (df[col] >= min_score).all()


class TestSaveTuningResults:
    """Tests for save_tuning_results."""

    def test_file_exists(self, grid_search_result: GridSearchCV, tmp_path: Any) -> None:
        path = save_tuning_results(grid_search_result, tmp_path, "tuning_results.json")
        assert path.exists()

    def test_json_has_best_params(self, grid_search_result: GridSearchCV, tmp_path: Any) -> None:
        path = save_tuning_results(grid_search_result, tmp_path, "tuning_results.json")
        with open(path) as f:
            data = json.load(f)
        assert "best_params" in data

    def test_no_numpy_types(self, grid_search_result: GridSearchCV, tmp_path: Any) -> None:
        path = save_tuning_results(grid_search_result, tmp_path, "tuning_results.json")
        with open(path) as f:
            raw = f.read()
        # Re-parse to verify it's valid JSON (no numpy serialization errors)
        data = json.loads(raw)
        assert isinstance(data, dict)
        # Ensure best_params values are plain Python types
        for value in data["best_params"].values():
            assert not isinstance(value, np.integer | np.floating | np.ndarray)

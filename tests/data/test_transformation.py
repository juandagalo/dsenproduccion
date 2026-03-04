"""Tests for custom transformers and feature pipeline builder."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from data.transformation import (
    NewCustomerTransformer,
    ServiceLevelConsolidator,
    build_feature_pipeline,
)

EXPECTED_OUTPUT_COLUMNS = 22


@pytest.fixture()
def service_df() -> pd.DataFrame:
    """Small DataFrame with service-level values to consolidate."""
    return pd.DataFrame(
        {
            "MultipleLines": pd.Categorical(["Yes", "No", "No phone service", None]),
            "OnlineSecurity": pd.Categorical(["Yes", "No internet service", "No", None]),
        }
    )


@pytest.fixture()
def tenure_df() -> pd.DataFrame:
    """Small DataFrame with tenure and MonthlyCharges columns."""
    return pd.DataFrame(
        {
            "tenure": [1, 6, 7, 24, None],
            "MonthlyCharges": [29.85, 56.95, 53.85, 70.70, 45.00],
        }
    )


@pytest.fixture()
def primary_df() -> pd.DataFrame:
    """DataFrame mimicking the primary churn schema with all required columns."""
    n = 20
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "gender": pd.Categorical(rng.choice(["Male", "Female"], n)),
            "SeniorCitizen": rng.choice([True, False], n),
            "Partner": rng.choice([True, False], n),
            "Dependents": rng.choice([True, False], n),
            "tenure": pd.array(rng.integers(0, 72, n), dtype="Int16"),
            "PhoneService": pd.Categorical(rng.choice(["Yes", "No"], n)),
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
            "TotalCharges": rng.uniform(18.0, 8700.0, n).round(2),
            "Churn": rng.choice([True, False], n),
        }
    )


class TestServiceLevelConsolidator:
    """Tests for ServiceLevelConsolidator transformer."""

    def test_replaces_service_levels(self, service_df: pd.DataFrame) -> None:
        transformer = ServiceLevelConsolidator()
        result = transformer.fit_transform(service_df)
        assert "No phone service" not in result["MultipleLines"].values
        assert "No internet service" not in result["OnlineSecurity"].values

    def test_preserves_nan(self, service_df: pd.DataFrame) -> None:
        transformer = ServiceLevelConsolidator()
        result = transformer.fit_transform(service_df)
        assert result["MultipleLines"].isna().sum() == 1
        assert result["OnlineSecurity"].isna().sum() == 1

    def test_custom_replacements(self, service_df: pd.DataFrame) -> None:
        custom = {"Yes": "Affirmative"}
        transformer = ServiceLevelConsolidator(replacements=custom)
        result = transformer.fit_transform(service_df)
        assert "Yes" not in result["MultipleLines"].values
        assert "Affirmative" in result["MultipleLines"].values

    def test_get_feature_names_out(self) -> None:
        transformer = ServiceLevelConsolidator()
        names = transformer.get_feature_names_out(["col_a", "col_b"])
        assert list(names) == ["col_a", "col_b"]

    def test_get_feature_names_out_after_fit(self, service_df: pd.DataFrame) -> None:
        transformer = ServiceLevelConsolidator()
        transformer.fit(service_df)
        names = transformer.get_feature_names_out()
        assert list(names) == ["MultipleLines", "OnlineSecurity"]


class TestNewCustomerTransformer:
    """Tests for NewCustomerTransformer."""

    def test_threshold_logic(self, tenure_df: pd.DataFrame) -> None:
        transformer = NewCustomerTransformer(threshold=6)
        result = transformer.fit_transform(tenure_df)
        expected = [1.0, 1.0, 0.0, 0.0, 1.0]  # NaN tenure -> 0 -> new
        assert list(result["is_new_customer"]) == expected

    def test_nan_handling(self, tenure_df: pd.DataFrame) -> None:
        transformer = NewCustomerTransformer(threshold=6)
        result = transformer.fit_transform(tenure_df)
        assert result["is_new_customer"].isna().sum() == 0

    def test_output_shape(self, tenure_df: pd.DataFrame) -> None:
        transformer = NewCustomerTransformer()
        result = transformer.fit_transform(tenure_df)
        assert result.shape == (5, 1)
        assert list(result.columns) == ["is_new_customer"]

    def test_get_feature_names_out(self) -> None:
        transformer = NewCustomerTransformer()
        names = transformer.get_feature_names_out()
        assert list(names) == ["is_new_customer"]


class TestBuildFeaturePipeline:
    """Tests for build_feature_pipeline function."""

    @pytest.fixture()
    def config(self) -> dict:  # type: ignore[type-arg]
        """Minimal config dict for testing."""
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

    def test_returns_column_transformer(self, config: dict) -> None:  # type: ignore[type-arg]
        pipeline = build_feature_pipeline(config)
        assert isinstance(pipeline, ColumnTransformer)

    def test_output_shape(
        self,
        config: dict,
        primary_df: pd.DataFrame,  # type: ignore[type-arg]
    ) -> None:
        pipeline = build_feature_pipeline(config)
        X = primary_df.drop(columns=["Churn"])
        result = pipeline.fit_transform(X)
        assert result.shape[0] == len(X)
        assert result.shape[1] == EXPECTED_OUTPUT_COLUMNS

    def test_zero_nan(
        self,
        config: dict,
        primary_df: pd.DataFrame,  # type: ignore[type-arg]
    ) -> None:
        pipeline = build_feature_pipeline(config)
        X = primary_df.drop(columns=["Churn"])
        result = pipeline.fit_transform(X)
        assert result.isna().sum().sum() == 0

    def test_dataframe_output(
        self,
        config: dict,
        primary_df: pd.DataFrame,  # type: ignore[type-arg]
    ) -> None:
        pipeline = build_feature_pipeline(config)
        X = primary_df.drop(columns=["Churn"])
        result = pipeline.fit_transform(X)
        assert isinstance(result, pd.DataFrame)

    def test_joblib_serializable(
        self,
        config: dict,
        primary_df: pd.DataFrame,
        tmp_path: pd.Series,  # type: ignore[type-arg]
    ) -> None:
        pipeline = build_feature_pipeline(config)
        X = primary_df.drop(columns=["Churn"])
        pipeline.fit(X)
        path = tmp_path / "pipeline.joblib"
        joblib.dump(pipeline, path)
        loaded = joblib.load(path)
        result = loaded.transform(X)
        assert result.shape[1] == EXPECTED_OUTPUT_COLUMNS

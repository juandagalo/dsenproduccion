"""Tests for data validation module."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]

from data.validation import (
    load_validation_config,
    validate_categories,
    validate_columns,
    validate_dataframe,
    validate_nulls,
    validate_numeric_ranges,
)


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
def validation_config() -> dict[str, Any]:
    """Load default validation configuration."""
    config: dict[str, Any] = load_validation_config()
    return config


class TestLoadValidationConfig:
    """Tests for load_validation_config."""

    def test_loads_default_config(self, validation_config: dict[str, Any]) -> None:
        assert "expected_columns" in validation_config
        assert "numeric_ranges" in validation_config
        assert "max_null_fraction" in validation_config
        assert "allowed_values" in validation_config

    def test_loads_custom_path(self, tmp_path: Any) -> None:
        custom = {
            "expected_columns": ["col_a", "col_b"],
            "max_null_fraction": 0.1,
            "numeric_ranges": {},
            "allowed_values": {},
        }
        path = tmp_path / "custom_validation.yml"
        with open(path, "w") as f:
            yaml.safe_dump(custom, f)
        config = load_validation_config(path)
        assert config["expected_columns"] == ["col_a", "col_b"]

    def test_expected_keys(self, validation_config: dict[str, Any]) -> None:
        expected_keys = {
            "expected_columns",
            "numeric_ranges",
            "max_null_fraction",
            "allowed_values",
        }
        assert expected_keys.issubset(set(validation_config.keys()))


class TestValidateColumns:
    """Tests for validate_columns."""

    def test_pass_valid_columns(
        self, primary_df: pd.DataFrame, validation_config: dict[str, Any]
    ) -> None:
        errors = validate_columns(primary_df, validation_config)
        assert errors == []

    def test_missing_column(
        self, primary_df: pd.DataFrame, validation_config: dict[str, Any]
    ) -> None:
        df = primary_df.drop(columns=["tenure"])
        errors = validate_columns(df, validation_config)
        assert len(errors) == 1
        assert "Missing columns" in errors[0]
        assert "tenure" in errors[0]

    def test_extra_column(
        self, primary_df: pd.DataFrame, validation_config: dict[str, Any]
    ) -> None:
        df = primary_df.copy()
        df["ExtraCol"] = 1
        errors = validate_columns(df, validation_config)
        assert len(errors) == 1
        assert "Extra columns" in errors[0]
        assert "ExtraCol" in errors[0]


class TestValidateNulls:
    """Tests for validate_nulls."""

    def test_pass_no_nulls(
        self, primary_df: pd.DataFrame, validation_config: dict[str, Any]
    ) -> None:
        errors = validate_nulls(primary_df, validation_config)
        assert errors == []

    def test_fail_high_null_fraction(
        self, primary_df: pd.DataFrame, validation_config: dict[str, Any]
    ) -> None:
        df = primary_df.copy()
        df.loc[:, "tenure"] = pd.array([pd.NA] * len(df), dtype="Int16")
        errors = validate_nulls(df, validation_config)
        assert len(errors) >= 1
        assert any("tenure" in e for e in errors)


class TestValidateNumericRanges:
    """Tests for validate_numeric_ranges."""

    def test_pass_in_range(
        self, primary_df: pd.DataFrame, validation_config: dict[str, Any]
    ) -> None:
        errors = validate_numeric_ranges(primary_df, validation_config)
        assert errors == []

    def test_fail_out_of_range(
        self, primary_df: pd.DataFrame, validation_config: dict[str, Any]
    ) -> None:
        df = primary_df.copy()
        df.loc[0, "MonthlyCharges"] = 999.0
        errors = validate_numeric_ranges(df, validation_config)
        assert len(errors) >= 1
        assert any("MonthlyCharges" in e for e in errors)


class TestValidateCategories:
    """Tests for validate_categories."""

    def test_pass_valid_values(
        self, primary_df: pd.DataFrame, validation_config: dict[str, Any]
    ) -> None:
        errors = validate_categories(primary_df, validation_config)
        assert errors == []

    def test_fail_invalid_value(
        self, primary_df: pd.DataFrame, validation_config: dict[str, Any]
    ) -> None:
        df = primary_df.copy()
        df["InternetService"] = df["InternetService"].cat.add_categories("Satellite")
        df.loc[0, "InternetService"] = "Satellite"
        errors = validate_categories(df, validation_config)
        assert len(errors) >= 1
        assert any("InternetService" in e for e in errors)


class TestValidateDataframe:
    """Tests for validate_dataframe."""

    def test_pass_clean_data(
        self, primary_df: pd.DataFrame, validation_config: dict[str, Any]
    ) -> None:
        errors = validate_dataframe(primary_df, validation_config)
        assert errors == []

    def test_fail_multiple_errors(
        self, primary_df: pd.DataFrame, validation_config: dict[str, Any]
    ) -> None:
        df = primary_df.drop(columns=["tenure"]).copy()
        df["InternetService"] = df["InternetService"].cat.add_categories("Satellite")
        df.loc[0, "InternetService"] = "Satellite"
        errors = validate_dataframe(df, validation_config)
        min_expected_errors = 2
        assert len(errors) >= min_expected_errors

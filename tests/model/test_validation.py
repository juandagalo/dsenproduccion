"""Tests for model validation module."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray

from model.validation import (
    load_model_validation_config,
    validate_class_distribution,
    validate_no_data_leakage,
    validate_overfitting,
    validate_performance_threshold,
)


@pytest.fixture()
def validation_config() -> dict[str, Any]:
    """Inline model validation config for testing."""
    return {
        "max_class_drift": 0.05,
        "max_overfit_gap": 0.10,
        "min_roc_auc": 0.70,
        "primary_metric": "roc_auc",
    }


@pytest.fixture()
def balanced_split() -> tuple[NDArray[np.integer[Any]], NDArray[np.integer[Any]]]:
    """Train/test labels with matching ~30% positive rate."""
    rng = np.random.default_rng(42)
    y_train = rng.choice([0, 1], size=100, p=[0.7, 0.3])
    y_test = rng.choice([0, 1], size=50, p=[0.7, 0.3])
    return y_train, y_test


@pytest.fixture()
def good_cv_results() -> dict[str, dict[str, NDArray[np.floating[Any]]]]:
    """CV results for 2 models with small train-test gaps."""
    return {
        "LogisticRegression": {
            "test_roc_auc": np.array([0.82, 0.84, 0.83]),
            "train_roc_auc": np.array([0.85, 0.86, 0.84]),
            "test_f1": np.array([0.70, 0.72, 0.71]),
            "train_f1": np.array([0.73, 0.74, 0.72]),
        },
        "GradientBoosting": {
            "test_roc_auc": np.array([0.80, 0.81, 0.79]),
            "train_roc_auc": np.array([0.83, 0.84, 0.82]),
            "test_f1": np.array([0.68, 0.70, 0.69]),
            "train_f1": np.array([0.71, 0.72, 0.70]),
        },
    }


class TestLoadModelValidationConfig:
    """Tests for load_model_validation_config."""

    def test_loads_default_config(self) -> None:
        config = load_model_validation_config()
        assert "max_class_drift" in config

    def test_loads_custom_path(self, tmp_path: Any) -> None:
        expected_drift = 0.10
        custom = {
            "max_class_drift": expected_drift,
            "max_overfit_gap": 0.20,
            "min_roc_auc": 0.60,
            "primary_metric": "roc_auc",
        }
        path = tmp_path / "custom_validation.yml"
        with open(path, "w") as f:
            yaml.safe_dump(custom, f)
        config = load_model_validation_config(path)
        assert config["max_class_drift"] == expected_drift

    def test_expected_keys(self) -> None:
        config = load_model_validation_config()
        expected_keys = {"max_class_drift", "max_overfit_gap", "min_roc_auc", "primary_metric"}
        assert expected_keys.issubset(config.keys())


class TestValidateClassDistribution:
    """Tests for validate_class_distribution."""

    def test_pass_balanced(
        self,
        balanced_split: tuple[NDArray[np.integer[Any]], NDArray[np.integer[Any]]],
        validation_config: dict[str, Any],
    ) -> None:
        y_train, y_test = balanced_split
        errors = validate_class_distribution(y_train, y_test, validation_config)
        assert errors == []

    def test_fail_drifted(self, validation_config: dict[str, Any]) -> None:
        y_train = np.array([0] * 75 + [1] * 25)
        y_test = np.array([0] * 20 + [1] * 30)
        errors = validate_class_distribution(y_train, y_test, validation_config)
        assert len(errors) > 0
        assert "drift" in errors[0]

    def test_pass_within_tolerance(
        self,
        validation_config: dict[str, Any],
    ) -> None:
        y_train = np.array([0] * 75 + [1] * 25)
        y_test = np.array([0] * 20 + [1] * 30)
        validation_config["max_class_drift"] = 0.50
        errors = validate_class_distribution(y_train, y_test, validation_config)
        assert errors == []


class TestValidateNoDataLeakage:
    """Tests for validate_no_data_leakage."""

    def test_pass_no_overlap(self) -> None:
        X_train = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        X_test = pd.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]})
        errors = validate_no_data_leakage(X_train, X_test)
        assert errors == []

    def test_fail_with_overlap(self) -> None:
        X_train = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        X_test = pd.DataFrame({"a": [2, 8, 9], "b": [5, 11, 12]})
        errors = validate_no_data_leakage(X_train, X_test)
        assert len(errors) == 1
        assert "leakage" in errors[0].lower()

    def test_works_with_numpy(self) -> None:
        X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_test = np.array([[5.0, 6.0], [7.0, 8.0]])
        errors = validate_no_data_leakage(X_train, X_test)
        assert errors == []


class TestValidateOverfitting:
    """Tests for validate_overfitting."""

    def test_pass_small_gap(
        self,
        good_cv_results: dict[str, dict[str, NDArray[np.floating[Any]]]],
        validation_config: dict[str, Any],
    ) -> None:
        errors = validate_overfitting(good_cv_results, validation_config)
        assert errors == []

    def test_fail_large_gap(self, validation_config: dict[str, Any]) -> None:
        cv_results: dict[str, dict[str, NDArray[np.floating[Any]]]] = {
            "OverfitModel": {
                "test_roc_auc": np.array([0.70, 0.70, 0.70]),
                "train_roc_auc": np.array([0.99, 0.99, 0.99]),
            },
        }
        errors = validate_overfitting(cv_results, validation_config)
        assert len(errors) == 1
        assert "gap" in errors[0]

    def test_ignores_negative_gap(self, validation_config: dict[str, Any]) -> None:
        cv_results: dict[str, dict[str, NDArray[np.floating[Any]]]] = {
            "GoodModel": {
                "test_roc_auc": np.array([0.90, 0.90, 0.90]),
                "train_roc_auc": np.array([0.80, 0.80, 0.80]),
            },
        }
        errors = validate_overfitting(cv_results, validation_config)
        assert errors == []


class TestValidatePerformanceThreshold:
    """Tests for validate_performance_threshold."""

    def test_pass_above_threshold(
        self,
        good_cv_results: dict[str, dict[str, NDArray[np.floating[Any]]]],
        validation_config: dict[str, Any],
    ) -> None:
        errors = validate_performance_threshold(good_cv_results, validation_config)
        assert errors == []

    def test_fail_below_threshold(self, validation_config: dict[str, Any]) -> None:
        cv_results: dict[str, dict[str, NDArray[np.floating[Any]]]] = {
            "WeakModel": {
                "test_roc_auc": np.array([0.60, 0.60, 0.60]),
                "train_roc_auc": np.array([0.65, 0.65, 0.65]),
            },
        }
        errors = validate_performance_threshold(cv_results, validation_config)
        assert len(errors) == 1
        assert "below" in errors[0].lower()

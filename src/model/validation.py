"""Model validation functions for train/test split and performance checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray


def load_model_validation_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load model validation configuration from YAML.

    Parameters
    ----------
    config_path : str, Path, or None
        Path to YAML config file. Defaults to ``conf/model_validation/churn.yml``.

    Returns
    -------
    dict[str, Any]
        Parsed model validation configuration.
    """
    if config_path is None:
        config_path = Path("conf/model_validation/churn.yml")
    config_path = Path(config_path)
    with open(config_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def validate_class_distribution(
    y_train: NDArray[np.integer[Any]],
    y_test: NDArray[np.integer[Any]],
    config: dict[str, Any],
) -> list[str]:
    """Check that class proportions between train and test sets are within tolerance.

    Parameters
    ----------
    y_train : NDArray[np.integer]
        Training set labels.
    y_test : NDArray[np.integer]
        Test set labels.
    config : dict[str, Any]
        Validation config with ``max_class_drift`` key.

    Returns
    -------
    list[str]
        Error messages for classes exceeding the drift threshold. Empty list if valid.
    """
    errors: list[str] = []
    max_drift = config["max_class_drift"]
    all_classes = np.unique(np.concatenate([y_train, y_test]))

    for cls in all_classes:
        train_prop = float(np.mean(y_train == cls))
        test_prop = float(np.mean(y_test == cls))
        drift = abs(train_prop - test_prop)

        if drift > max_drift:
            errors.append(
                f"Class {cls}: train proportion {train_prop:.4f} vs test {test_prop:.4f}"
                f" (drift {drift:.4f} > {max_drift})"
            )

    return errors


def validate_no_data_leakage(
    X_train: NDArray[np.floating[Any]] | pd.DataFrame,
    X_test: NDArray[np.floating[Any]] | pd.DataFrame,
) -> list[str]:
    """Check that no rows are shared between train and test sets.

    Parameters
    ----------
    X_train : NDArray[np.floating] or pd.DataFrame
        Training set features.
    X_test : NDArray[np.floating] or pd.DataFrame
        Test set features.

    Returns
    -------
    list[str]
        Error messages if overlapping rows are found. Empty list if valid.
    """
    errors: list[str] = []

    if isinstance(X_train, pd.DataFrame):
        train_rows = set(X_train.itertuples(index=False, name=None))
    else:
        train_rows = {tuple(row) for row in X_train}

    if isinstance(X_test, pd.DataFrame):
        test_rows = set(X_test.itertuples(index=False, name=None))
    else:
        test_rows = {tuple(row) for row in X_test}

    overlap = train_rows & test_rows
    if overlap:
        errors.append(f"Data leakage detected: {len(overlap)} rows shared between train and test")

    return errors


def validate_overfitting(
    cv_results: dict[str, dict[str, NDArray[np.floating[Any]]]],
    config: dict[str, Any],
) -> list[str]:
    """Check that train-test performance gaps do not exceed the configured threshold.

    Parameters
    ----------
    cv_results : dict[str, dict[str, NDArray[np.floating]]]
        Cross-validation results mapping model names to metric dictionaries.
        Each metric dict has ``train_*`` and ``test_*`` keys with NDArray values.
    config : dict[str, Any]
        Validation config with ``max_overfit_gap`` key.

    Returns
    -------
    list[str]
        Error messages for model/metric pairs exceeding the gap. Empty list if valid.
    """
    errors: list[str] = []
    max_gap = config["max_overfit_gap"]

    for model_name, metrics in cv_results.items():
        train_keys = [k for k in metrics if k.startswith("train_")]
        for train_key in train_keys:
            metric_name = train_key.removeprefix("train_")
            test_key = f"test_{metric_name}"

            if test_key not in metrics:
                continue

            train_mean = float(np.mean(metrics[train_key]))
            test_mean = float(np.mean(metrics[test_key]))
            gap = train_mean - test_mean

            if gap > max_gap:
                errors.append(
                    f"{model_name} - {metric_name}: train {train_mean:.4f} vs"
                    f" test {test_mean:.4f} (gap {gap:.4f} > {max_gap})"
                )

    return errors


def validate_performance_threshold(
    cv_results: dict[str, dict[str, NDArray[np.floating[Any]]]],
    config: dict[str, Any],
) -> list[str]:
    """Check that the best model meets the minimum performance threshold.

    Parameters
    ----------
    cv_results : dict[str, dict[str, NDArray[np.floating]]]
        Cross-validation results mapping model names to metric dictionaries.
    config : dict[str, Any]
        Validation config with ``primary_metric`` and ``min_{metric}`` keys.

    Returns
    -------
    list[str]
        Error messages if best model is below threshold. Empty list if valid.
    """
    errors: list[str] = []
    primary_metric = config["primary_metric"]
    test_key = f"test_{primary_metric}"
    min_key = f"min_{primary_metric}"
    min_threshold = config[min_key]

    best_model = None
    best_score = -np.inf

    for model_name, metrics in cv_results.items():
        if test_key not in metrics:
            continue
        mean_score = float(np.mean(metrics[test_key]))
        if mean_score > best_score:
            best_score = mean_score
            best_model = model_name

    if best_model is not None and best_score < min_threshold:
        errors.append(
            f"Best model '{best_model}' {primary_metric}={best_score:.4f}"
            f" is below minimum threshold {min_threshold}"
        )

    return errors

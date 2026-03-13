"""Data validation functions for config-driven schema checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml  # type: ignore[import-untyped]


def load_validation_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load data validation configuration from YAML.

    Parameters
    ----------
    config_path : str, Path, or None
        Path to YAML config file. Defaults to ``conf/data_validation/churn.yml``.

    Returns
    -------
    dict[str, Any]
        Parsed validation configuration.
    """
    if config_path is None:
        config_path = Path("conf/data_validation/churn.yml")
    config_path = Path(config_path)
    with open(config_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def validate_columns(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    """Check that DataFrame columns match the expected set.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    config : dict[str, Any]
        Validation config with ``expected_columns`` key.

    Returns
    -------
    list[str]
        Error messages for missing or extra columns. Empty list if valid.
    """
    errors: list[str] = []
    expected = set(config["expected_columns"])
    actual = set(df.columns)

    missing = expected - actual
    extra = actual - expected

    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")
    if extra:
        errors.append(f"Extra columns: {sorted(extra)}")

    return errors


def validate_nulls(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    """Check that no column exceeds the maximum allowed null fraction.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    config : dict[str, Any]
        Validation config with ``max_null_fraction`` key.

    Returns
    -------
    list[str]
        Error messages for columns exceeding the null threshold. Empty list if valid.
    """
    errors: list[str] = []
    max_fraction = config["max_null_fraction"]

    for col in df.columns:
        null_fraction = df[col].isna().mean()
        if null_fraction > max_fraction:
            errors.append(
                f"Column '{col}' has {null_fraction:.2%} nulls (max allowed: {max_fraction:.2%})"
            )

    return errors


def validate_numeric_ranges(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    """Check that numeric columns fall within configured min/max ranges.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    config : dict[str, Any]
        Validation config with ``numeric_ranges`` key.

    Returns
    -------
    list[str]
        Error messages for out-of-range values. Empty list if valid.
    """
    errors: list[str] = []
    ranges = config.get("numeric_ranges", {})

    for col, bounds in ranges.items():
        if col not in df.columns:
            continue

        values = df[col].dropna()
        col_min = values.min()
        col_max = values.max()

        if col_min < bounds["min"]:
            errors.append(
                f"Column '{col}' has min value {col_min}, below allowed minimum {bounds['min']}"
            )
        if col_max > bounds["max"]:
            errors.append(
                f"Column '{col}' has max value {col_max}, above allowed maximum {bounds['max']}"
            )

    return errors


def validate_categories(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    """Check that categorical columns contain only allowed values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    config : dict[str, Any]
        Validation config with ``allowed_values`` key.

    Returns
    -------
    list[str]
        Error messages for unexpected category values. Empty list if valid.
    """
    errors: list[str] = []
    allowed_values = config.get("allowed_values", {})

    for col, allowed in allowed_values.items():
        if col not in df.columns:
            continue

        actual = set(df[col].dropna().unique())
        invalid = actual - set(allowed)

        if invalid:
            errors.append(f"Column '{col}' has invalid values: {sorted(invalid)}")

    return errors


def validate_dataframe(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    """Run all validation checks on a DataFrame.

    Orchestrates column, null, numeric range, and category validation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    config : dict[str, Any]
        Full validation configuration dictionary.

    Returns
    -------
    list[str]
        Combined error messages from all validators. Empty list if all pass.
    """
    errors: list[str] = []
    errors.extend(validate_columns(df, config))
    errors.extend(validate_nulls(df, config))
    errors.extend(validate_numeric_ranges(df, config))
    errors.extend(validate_categories(df, config))
    return errors

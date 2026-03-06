"""Inference module for churn prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import pandas as pd
import yaml  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline


def load_serving_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load serving configuration from YAML.

    Parameters
    ----------
    config_path : str, Path, or None
        Path to YAML config file. Defaults to ``conf/model_serving/churn.yml``.

    Returns
    -------
    dict[str, Any]
        Parsed serving configuration.
    """
    if config_path is None:
        config_path = Path("conf/model_serving/churn.yml")
    config_path = Path(config_path)
    with open(config_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def load_model(model_path: str | Path) -> Pipeline:
    """Load a trained sklearn pipeline from disk.

    Parameters
    ----------
    model_path : str or Path
        Path to the joblib file.

    Returns
    -------
    Pipeline
        Fitted sklearn pipeline.
    """
    pipeline: Pipeline = joblib.load(model_path)
    return pipeline


def load_evaluation_results(results_path: str | Path) -> dict[str, Any]:
    """Load evaluation results from JSON.

    Parameters
    ----------
    results_path : str or Path
        Path to the evaluation results JSON file.

    Returns
    -------
    dict[str, Any]
        Evaluation results including threshold, metrics, and coefficients.
    """
    results_path = Path(results_path)
    with open(results_path) as f:
        results: dict[str, Any] = json.load(f)
    return results


def build_customer_dataframe(customer_data: dict[str, Any]) -> pd.DataFrame:
    """Build a single-row DataFrame from customer data dictionary.

    Casts boolean and categorical columns to the expected dtypes
    for the sklearn pipeline.

    Parameters
    ----------
    customer_data : dict[str, Any]
        Dictionary with 16 feature keys matching the primary schema.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame ready for pipeline prediction.
    """
    df = pd.DataFrame([customer_data])

    bool_cols = ["SeniorCitizen", "Partner", "Dependents", "PaperlessBilling"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    cat_cols = [
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "InternetService",
        "PaymentMethod",
        "Contract",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def predict_churn(
    pipeline: Pipeline,
    df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Add churn probability and prediction columns to a DataFrame.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn pipeline with ``predict_proba`` method.
    df : pd.DataFrame
        Input DataFrame with features matching the pipeline's expected schema.
    threshold : float
        Classification threshold for the positive class.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``churn_probability`` and ``churn_predicted``
        columns added.
    """
    result = df.copy()
    probas = pipeline.predict_proba(df)[:, 1]
    result["churn_probability"] = probas
    result["churn_predicted"] = (probas >= threshold).astype(int)
    return result


def get_risk_level(
    probability: float,
    low_max: float = 0.3,
    medium_max: float = 0.6,
) -> str:
    """Classify a churn probability into a risk level.

    Parameters
    ----------
    probability : float
        Churn probability (0 to 1).
    low_max : float
        Upper bound for "Low" risk (exclusive). Default 0.3.
    medium_max : float
        Upper bound for "Medium" risk (exclusive). Default 0.6.

    Returns
    -------
    str
        One of "Low", "Medium", or "High".
    """
    if probability < low_max:
        return "Low"
    if probability < medium_max:
        return "Medium"
    return "High"

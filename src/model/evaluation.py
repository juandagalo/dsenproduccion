"""Threshold optimization and model interpretation module for churn evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline


def load_evaluation_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load evaluation configuration from YAML.

    Parameters
    ----------
    config_path : str, Path, or None
        Path to YAML config file. Defaults to ``conf/model_evaluation/churn.yml``.

    Returns
    -------
    dict[str, Any]
        Parsed evaluation configuration.
    """
    if config_path is None:
        config_path = Path("conf/model_evaluation/churn.yml")
    config_path = Path(config_path)
    with open(config_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def compute_threshold_metrics(
    y_true: NDArray[np.integer[Any]],
    y_proba: NDArray[np.floating[Any]],
    thresholds: NDArray[np.floating[Any]] | None = None,
) -> pd.DataFrame:
    """Compute precision, recall, f1, and confusion matrix values for each threshold.

    Parameters
    ----------
    y_true : NDArray[np.integer]
        Ground-truth binary labels.
    y_proba : NDArray[np.floating]
        Predicted probabilities for the positive class.
    thresholds : NDArray[np.floating] or None
        Array of thresholds to evaluate. Defaults to ``np.arange(0.01, 0.99, 0.01)``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: threshold, precision, recall, f1, tp, fp, fn, tn.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)

    rows: list[dict[str, Any]] = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append(
            {
                "threshold": float(t),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }
        )
    return pd.DataFrame(rows)


def find_optimal_threshold(
    metrics_df: pd.DataFrame,
    metric: str = "f1",
) -> tuple[float, float]:
    """Find the threshold that maximizes a given metric.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of ``compute_threshold_metrics``.
    metric : str
        Column name in ``metrics_df`` to maximize.

    Returns
    -------
    tuple[float, float]
        A tuple of ``(threshold, metric_value)`` at the optimal point.
    """
    best_idx = metrics_df[metric].idxmax()
    best_row = metrics_df.loc[best_idx]
    return float(best_row["threshold"]), float(best_row[metric])


def extract_feature_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """Extract logistic regression coefficients and feature names from a pipeline.

    The feature names are obtained from the scaler's ``feature_names_in_``
    attribute (set when the preprocessor outputs a DataFrame via ``set_output``).
    If no scaler is present, falls back to ``preprocessor.get_feature_names_out()``.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn pipeline with ``preprocessor`` and ``model`` named steps.
        Optionally contains a ``scaler`` step between preprocessor and model.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: feature, coefficient. Sorted by absolute
        coefficient value descending.
    """
    model = pipeline.named_steps["model"]
    coefficients = model.coef_[0]

    if "scaler" in pipeline.named_steps:
        feature_names = pipeline.named_steps["scaler"].feature_names_in_
    else:
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    coef_df = coef_df.sort_values(
        "coefficient", key=lambda x: x.abs(), ascending=False
    ).reset_index(drop=True)
    return coef_df


def build_threshold_tradeoff_table(
    metrics_df: pd.DataFrame,
    candidate_thresholds: list[float],
) -> pd.DataFrame:
    """Filter metrics to rows closest to each candidate threshold.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of ``compute_threshold_metrics``.
    candidate_thresholds : list[float]
        List of desired threshold values to extract.

    Returns
    -------
    pd.DataFrame
        Subset of ``metrics_df`` with one row per candidate threshold.
    """
    indices: list[int] = []
    threshold_values = metrics_df["threshold"].values
    for candidate in candidate_thresholds:
        idx = int(np.argmin(np.abs(threshold_values - candidate)))
        indices.append(idx)
    return metrics_df.iloc[indices].reset_index(drop=True)


def classify_with_threshold(
    y_proba: NDArray[np.floating[Any]],
    threshold: float,
) -> NDArray[np.integer[Any]]:
    """Classify probabilities using a custom threshold.

    Parameters
    ----------
    y_proba : NDArray[np.floating]
        Predicted probabilities for the positive class.
    threshold : float
        Classification threshold.

    Returns
    -------
    NDArray[np.integer]
        Binary predictions (0 or 1).
    """
    return (y_proba >= threshold).astype(int)


def save_evaluation_results(
    results: dict[str, Any],
    output_path: str | Path,
    filename: str,
) -> Path:
    """Save evaluation results as JSON.

    Numpy arrays and types are converted to Python native types for
    JSON serialization.

    Parameters
    ----------
    results : dict[str, Any]
        Evaluation results dictionary to save.
    output_path : str or Path
        Directory to save results in.
    filename : str
        Filename for the JSON file.

    Returns
    -------
    Path
        Full path to saved JSON file.
    """
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    serializable = _convert_numpy(results)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    return path


def _convert_numpy(value: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _convert_numpy(v) for k, v in value.items()}
    return [_convert_numpy(item) for item in value] if isinstance(value, list) else value

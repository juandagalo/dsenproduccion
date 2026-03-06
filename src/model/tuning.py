"""Hyperparameter tuning module for churn model.

Provides functions to run GridSearchCV on the full sklearn pipeline,
summarize results, compare against baseline, and save tuning artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from model.training import build_cv_strategy


def load_tuning_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load tuning configuration from YAML.

    Parameters
    ----------
    config_path : str, Path, or None
        Path to YAML config file. Defaults to ``conf/model_train/churn_tuning.yml``.

    Returns
    -------
    dict[str, Any]
        Parsed tuning configuration.
    """
    if config_path is None:
        config_path = Path("conf/model_train/churn_tuning.yml")
    config_path = Path(config_path)
    with open(config_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def build_param_grid(
    tuning_config: dict[str, Any],
) -> list[dict[str, list[Any]]]:
    """Build parameter grid list from config for GridSearchCV.

    Parameters
    ----------
    tuning_config : dict[str, Any]
        Tuning configuration containing ``param_grid`` key.

    Returns
    -------
    list[dict[str, list[Any]]]
        List of parameter dictionaries for ``GridSearchCV``.
    """
    grid: list[dict[str, list[Any]]] = tuning_config["param_grid"]
    return grid


def run_grid_search(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,  # type: ignore[type-arg]
    param_grid: list[dict[str, list[Any]]],
    tuning_config: dict[str, Any],
) -> GridSearchCV:
    """Run GridSearchCV on the full pipeline (preprocessor included = no data leakage).

    The entire sklearn ``Pipeline`` (preprocessor + scaler + model) is passed to
    ``GridSearchCV``, so the preprocessor is fitted only on each fold's training
    data, preventing any data leakage.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted sklearn pipeline with preprocessor, optional scaler, and model.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    param_grid : list[dict[str, list[Any]]]
        Parameter grid for ``GridSearchCV``.
    tuning_config : dict[str, Any]
        Tuning configuration with ``cv``, ``scoring``, and ``refit`` keys.

    Returns
    -------
    GridSearchCV
        Fitted ``GridSearchCV`` object.
    """
    cv_strategy = build_cv_strategy(tuning_config["cv"])
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=tuning_config["scoring"],
        cv=cv_strategy,
        refit=tuning_config.get("refit", True),
        return_train_score=True,
        n_jobs=-1,
    )
    grid_search.fit(X, y)
    return grid_search


def summarize_grid_search(grid_search: GridSearchCV) -> pd.DataFrame:
    """Extract cv_results_ into a clean DataFrame sorted by rank.

    Parameters
    ----------
    grid_search : GridSearchCV
        Fitted ``GridSearchCV`` object.

    Returns
    -------
    pd.DataFrame
        Summary with columns: ``params``, ``mean_test_score``,
        ``std_test_score``, ``mean_train_score``, ``std_train_score``,
        ``rank_test_score``.
    """
    results = pd.DataFrame(grid_search.cv_results_)
    columns = [
        "params",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "rank_test_score",
    ]
    summary = results[columns].sort_values("rank_test_score")
    return summary


def compare_baseline_vs_tuned(
    baseline_cv_results: dict[str, dict[str, Any]],
    baseline_model_name: str,
    grid_search: GridSearchCV,
    scoring: str,
) -> pd.DataFrame:
    """Compare baseline CV results vs tuned model best score.

    Parameters
    ----------
    baseline_cv_results : dict[str, dict[str, Any]]
        Baseline cross-validation results keyed by model name.
    baseline_model_name : str
        Name of the baseline model to compare against.
    grid_search : GridSearchCV
        Fitted ``GridSearchCV`` object with tuned results.
    scoring : str
        Scoring metric name without ``test_`` prefix (e.g. ``"roc_auc"``).

    Returns
    -------
    pd.DataFrame
        DataFrame with rows ``[baseline, tuned]`` and columns
        ``[mean_<scoring>, std_<scoring>]``.
    """
    baseline_key = f"test_{scoring}"
    baseline_scores = np.array(baseline_cv_results[baseline_model_name][baseline_key])

    tuned_mean = grid_search.best_score_
    tuned_std = grid_search.cv_results_["std_test_score"][grid_search.best_index_]

    comparison = pd.DataFrame(
        {
            f"mean_{scoring}": [float(baseline_scores.mean()), tuned_mean],
            f"std_{scoring}": [float(baseline_scores.std()), tuned_std],
        },
        index=["baseline", "tuned"],
    )
    return comparison


def save_tuning_results(
    grid_search: GridSearchCV,
    output_path: str | Path,
    filename: str,
) -> Path:
    """Save GridSearchCV results as JSON.

    Numpy arrays and types are converted to Python native types for
    JSON serialization.

    Parameters
    ----------
    grid_search : GridSearchCV
        Fitted ``GridSearchCV`` object.
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

    results: dict[str, Any] = {
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
        "best_index": int(grid_search.best_index_),
        "all_results": [],
    }

    cv_results = grid_search.cv_results_
    for i in range(len(cv_results["params"])):
        entry: dict[str, Any] = {
            "params": {k: _convert(v) for k, v in cv_results["params"][i].items()},
            "mean_test_score": float(cv_results["mean_test_score"][i]),
            "std_test_score": float(cv_results["std_test_score"][i]),
            "mean_train_score": float(cv_results["mean_train_score"][i]),
            "std_train_score": float(cv_results["std_train_score"][i]),
            "rank_test_score": int(cv_results["rank_test_score"][i]),
        }
        results["all_results"].append(entry)

    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def _convert(value: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value

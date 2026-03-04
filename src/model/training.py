"""Core training module for churn baseline model training.

Provides functions to build sklearn pipelines, cross-validate multiple classifiers,
compare results, select the best model, and save artifacts.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from data.transformation import build_feature_pipeline

MODEL_REGISTRY: dict[str, type[ClassifierMixin]] = {
    "sklearn.linear_model.LogisticRegression": LogisticRegression,
    "sklearn.ensemble.RandomForestClassifier": RandomForestClassifier,
    "sklearn.ensemble.GradientBoostingClassifier": GradientBoostingClassifier,
}


def load_train_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load training configuration from YAML.

    Parameters
    ----------
    config_path : str, Path, or None
        Path to YAML config file. Defaults to ``conf/model_train/churn.yml``.

    Returns
    -------
    dict[str, Any]
        Parsed training configuration.
    """
    if config_path is None:
        config_path = Path("conf/model_train/churn.yml")
    config_path = Path(config_path)
    with open(config_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def build_model_pipelines(
    feature_config: dict[str, Any],
    train_config: dict[str, Any],
) -> dict[str, Pipeline]:
    """Build named sklearn pipelines for each model in the training config.

    Each pipeline gets a fresh ``ColumnTransformer`` from ``build_feature_pipeline``
    to avoid shared fitted state. Logistic regression pipelines include a
    ``StandardScaler`` between the preprocessor and classifier.

    Parameters
    ----------
    feature_config : dict[str, Any]
        Feature engineering configuration for ``build_feature_pipeline``.
    train_config : dict[str, Any]
        Training configuration containing model definitions.

    Returns
    -------
    dict[str, Pipeline]
        Mapping of model name to unfitted sklearn Pipeline.
    """
    pipelines: dict[str, Pipeline] = {}
    for name, model_cfg in train_config["models"].items():
        preprocessor = build_feature_pipeline(feature_config)
        model_class = MODEL_REGISTRY[model_cfg["class"]]
        model = model_class(**model_cfg.get("params", {}))

        steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
        if model_cfg.get("needs_scaler", False):
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", model))

        pipelines[name] = Pipeline(steps)
    return pipelines


def build_cv_strategy(cv_config: dict[str, Any]) -> StratifiedKFold:
    """Build a StratifiedKFold cross-validation strategy from config.

    Parameters
    ----------
    cv_config : dict[str, Any]
        Configuration with ``n_splits``, ``shuffle``, and ``random_state``.

    Returns
    -------
    StratifiedKFold
        Configured cross-validation splitter.
    """
    return StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=cv_config["shuffle"],
        random_state=cv_config["random_state"],
    )


def cross_validate_models(
    pipelines: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    train_config: dict[str, Any],
) -> dict[str, dict[str, NDArray[np.floating[Any]]]]:
    """Run cross-validation for each pipeline.

    For ``GradientBoostingClassifier`` pipelines, ``compute_sample_weight("balanced", y)``
    is passed via ``fit_params`` since GB lacks a ``class_weight`` parameter.

    Parameters
    ----------
    pipelines : dict[str, Pipeline]
        Named pipelines from ``build_model_pipelines``.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    train_config : dict[str, Any]
        Training config with ``cv`` and ``scoring`` sections.

    Returns
    -------
    dict[str, dict[str, NDArray]]
        Mapping of model name to ``cross_validate`` result dict.
    """
    cv = build_cv_strategy(train_config["cv"])
    scoring = train_config["scoring"]
    results: dict[str, dict[str, NDArray[np.floating[Any]]]] = {}

    for name, pipeline in pipelines.items():
        fit_params: dict[str, Any] = {}
        model_step = pipeline.named_steps["model"]
        if isinstance(model_step, GradientBoostingClassifier):
            weights = compute_sample_weight("balanced", y)
            fit_params["model__sample_weight"] = weights

        cv_result = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            params=fit_params if fit_params else None,
        )
        results[name] = cv_result
    return results


def compare_cv_results(
    cv_results: dict[str, dict[str, NDArray[np.floating[Any]]]],
    scoring: list[str],
) -> pd.DataFrame:
    """Build a comparison table of cross-validation results.

    Parameters
    ----------
    cv_results : dict
        Output of ``cross_validate_models``.
    scoring : list[str]
        List of scoring metric names.

    Returns
    -------
    pd.DataFrame
        Rows are models, columns are metric mean and std for train and test.
    """
    rows: list[dict[str, Any]] = []
    for model_name, result in cv_results.items():
        row: dict[str, Any] = {"model": model_name}
        for metric in scoring:
            for split in ("train", "test"):
                key = f"{split}_{metric}"
                values = result[key]
                row[f"{key}_mean"] = float(np.mean(values))
                row[f"{key}_std"] = float(np.std(values))
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def select_best_model(
    cv_results: dict[str, dict[str, NDArray[np.floating[Any]]]],
    primary_metric: str,
) -> str:
    """Select the model with the highest mean on the primary metric.

    Parameters
    ----------
    cv_results : dict
        Output of ``cross_validate_models``.
    primary_metric : str
        Key in each result dict to compare (e.g. ``"test_roc_auc"``).

    Returns
    -------
    str
        Name of the best model.
    """
    best_name = ""
    best_score = -np.inf
    for name, result in cv_results.items():
        mean_score = float(np.mean(result[primary_metric]))
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
    return best_name


def train_final_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    needs_sample_weight: bool = False,
) -> Pipeline:
    """Retrain a pipeline on the full dataset.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted or previously fitted pipeline (will be refit).
    X : pd.DataFrame
        Full feature matrix.
    y : pd.Series
        Full target vector.
    needs_sample_weight : bool
        If True, compute balanced sample weights for ``model__sample_weight``.

    Returns
    -------
    Pipeline
        Fitted pipeline.
    """
    fit_params: dict[str, Any] = {}
    if needs_sample_weight:
        weights = compute_sample_weight("balanced", y)
        fit_params["model__sample_weight"] = weights
    pipeline.fit(X, y, **fit_params)
    return pipeline


def save_model(pipeline: Pipeline, output_path: str | Path, filename: str) -> Path:
    """Save a fitted pipeline to disk using joblib.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn pipeline.
    output_path : str or Path
        Directory to save the model in.
    filename : str
        Filename for the saved model.

    Returns
    -------
    Path
        Full path to saved model file.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / filename
    joblib.dump(pipeline, filepath)
    return filepath


def save_cv_results(
    cv_results: dict[str, dict[str, NDArray[np.floating[Any]]]],
    best_model_name: str,
    output_path: str | Path,
    filename: str,
) -> Path:
    """Save cross-validation results to a JSON file.

    Converts numpy arrays to Python lists for JSON serialization.

    Parameters
    ----------
    cv_results : dict
        Output of ``cross_validate_models``.
    best_model_name : str
        Name of the selected best model.
    output_path : str or Path
        Directory to save results in.
    filename : str
        Filename for the JSON file.

    Returns
    -------
    Path
        Full path to saved JSON file.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / filename

    serializable: dict[str, Any] = {"best_model_name": best_model_name, "models": {}}
    for model_name, result in cv_results.items():
        model_data: dict[str, Any] = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                model_data[key] = value.tolist()
            else:
                model_data[key] = value
        serializable["models"][model_name] = model_data

    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)
    return filepath

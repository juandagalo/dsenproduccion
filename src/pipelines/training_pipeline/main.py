"""Training pipeline entry point.

Loads primary data, builds model pipelines, runs cross-validation,
selects the best model, retrains on full data, and saves artifacts.

Usage: PYTHONPATH=src uv run python -m pipelines.training_pipeline.main
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from data.transformation import load_feature_config
from model.training import (
    build_model_pipelines,
    compare_cv_results,
    cross_validate_models,
    load_train_config,
    save_cv_results,
    save_model,
    select_best_model,
    train_final_model,
)


def main() -> None:
    """Run the training pipeline end-to-end."""
    train_config = load_train_config()
    feature_config = load_feature_config(train_config.get("feature_config_path"))

    input_path = Path(train_config["input_path"])
    target = train_config["target"]

    print(f"Loading primary data from {input_path}")
    df = pd.read_parquet(input_path)
    print(f"  Shape: {df.shape}")

    X = df.drop(columns=[target])
    y = df[target].astype(int)
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  Class distribution:\n{y.value_counts().to_string()}")

    print("\nBuilding model pipelines...")
    pipelines = build_model_pipelines(feature_config, train_config)
    for name, pipeline in pipelines.items():
        print(f"  {name}: {[step_name for step_name, _ in pipeline.steps]}")

    print("\nRunning cross-validation...")
    cv_results = cross_validate_models(pipelines, X, y, train_config)

    print("\nComparison table:")
    comparison = compare_cv_results(cv_results, train_config["scoring"])
    print(comparison.to_string())

    primary_metric = train_config["primary_metric"]
    best_name = select_best_model(cv_results, primary_metric)
    best_score = cv_results[best_name][primary_metric].mean()
    print(f"\nBest model: {best_name} ({primary_metric} = {best_score:.4f})")

    print("\nRetraining best model on full data...")
    best_pipeline = pipelines[best_name]
    needs_sw = isinstance(best_pipeline.named_steps["model"], GradientBoostingClassifier)
    fitted_pipeline = train_final_model(best_pipeline, X, y, needs_sample_weight=needs_sw)

    preds = fitted_pipeline.predict(X.head())
    print(f"  Sanity check (first 5 predictions): {preds}")

    model_path = save_model(
        fitted_pipeline,
        train_config["model_output_path"],
        train_config["best_pipeline_filename"],
    )
    print(f"\nSaved model to {model_path}")

    results_path = save_cv_results(
        cv_results,
        best_name,
        train_config["reporting_output_path"],
        train_config["cv_results_filename"],
    )
    print(f"Saved CV results to {results_path}")


if __name__ == "__main__":
    main()

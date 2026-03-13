"""Inference pipeline entry point.

Loads primary data, applies the trained model, generates predictions
with optimized threshold, and saves output to the model output layer.

Usage: PYTHONPATH=src uv run python -m pipelines.inference_pipeline.main
"""

from pathlib import Path

import pandas as pd

from inference.predict import (
    get_risk_level,
    load_evaluation_results,
    load_model,
    load_serving_config,
    predict_churn,
)


def main() -> None:
    """Run the inference pipeline end-to-end."""
    config = load_serving_config()

    print("Loading evaluation results...")
    eval_results = load_evaluation_results(config["evaluation_results_path"])
    threshold: float = eval_results["chosen_threshold"]
    print(f"  Chosen threshold: {threshold}")

    print("Loading model...")
    pipeline = load_model(config["model_path"])
    print(f"  Pipeline steps: {[name for name, _ in pipeline.steps]}")

    input_path = Path(config["input_path"])
    print(f"\nLoading primary data from {input_path}")
    df = pd.read_parquet(input_path)
    print(f"  Shape: {df.shape}")

    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
        print("  Dropped 'Churn' column")

    print("\nGenerating predictions...")
    df = predict_churn(pipeline, df, threshold)
    print(f"  Churn rate: {df['churn_predicted'].mean():.2%}")

    low_max: float = config["risk_levels"]["low_max"]
    medium_max: float = config["risk_levels"]["medium_max"]
    df["risk_level"] = df["churn_probability"].apply(
        lambda p: get_risk_level(p, low_max, medium_max)
    )
    print(f"  Risk distribution:\n{df['risk_level'].value_counts().to_string()}")

    output_path = Path(config["prediction_output_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    predictions_path = output_path / "predictions.parquet"
    df.to_parquet(predictions_path, index=False)
    print(f"\nSaved predictions to {predictions_path}")
    print(f"  Output shape: {df.shape}")


if __name__ == "__main__":
    main()

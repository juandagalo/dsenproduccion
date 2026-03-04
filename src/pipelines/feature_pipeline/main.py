"""Feature pipeline entry point.

Loads primary data, builds the feature preprocessing pipeline,
transforms features, and saves output to the feature layer.

Usage: uv run python -m pipelines.feature_pipeline.main
"""

from pathlib import Path

import pandas as pd

from data.transformation import build_feature_pipeline, load_feature_config


def main() -> None:
    """Run the feature pipeline end-to-end."""
    config = load_feature_config()

    input_path = Path(config["input_path"])
    output_path = Path(config["feature_output_path"])
    target = config["target"]

    print(f"Loading primary data from {input_path}")
    df = pd.read_parquet(input_path)
    print(f"  Shape: {df.shape}")

    X = df.drop(columns=[target])
    y = df[target].astype(int)
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  Class distribution:\n{y.value_counts().to_string()}")

    print("\nBuilding feature pipeline...")
    pipeline = build_feature_pipeline(config)
    print(f"  Pipeline: {pipeline}")

    print("\nTransforming features (fit_transform on full data for inspection)...")
    X_transformed = pipeline.fit_transform(X)
    print(f"  Output shape: {X_transformed.shape}")
    print(f"  NaN count: {X_transformed.isna().sum().sum()}")
    print(f"  Columns: {list(X_transformed.columns)}")

    output_path.mkdir(parents=True, exist_ok=True)
    features_path = output_path / "features.parquet"
    target_path = output_path / "target.parquet"

    X_transformed.to_parquet(features_path, index=False)
    y.to_frame().to_parquet(target_path, index=False)
    print(f"\nSaved features to {features_path}")
    print(f"Saved target to {target_path}")


if __name__ == "__main__":
    main()

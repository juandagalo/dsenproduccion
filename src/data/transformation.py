"""Custom transformers and pipeline builder for churn feature engineering."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder


class ServiceLevelConsolidator(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Replace service-level values like 'No internet service' with 'No'.

    Converts category dtype to object before replacement to avoid FutureWarning.

    Parameters
    ----------
    replacements : dict[str, str]
        Mapping of values to replace, e.g. {"No internet service": "No"}.
    """

    def __init__(self, replacements: dict[str, str] | None = None) -> None:
        self.replacements = replacements or {
            "No internet service": "No",
            "No phone service": "No",
        }

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ServiceLevelConsolidator":
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)
        else:
            self.feature_names_in_ = np.arange(X.shape[1])  # type: ignore[union-attr]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in X_out.columns:
            if X_out[col].dtype.name == "category":
                X_out[col] = X_out[col].astype(object)
            X_out[col] = X_out[col].replace(self.replacements)
        return X_out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is not None:
            return np.asarray(input_features)
        return np.asarray(self.feature_names_in_)


class NewCustomerTransformer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Create binary is_new_customer feature from tenure column.

    Parameters
    ----------
    threshold : int
        Tenure in months below which a customer is considered new.
    tenure_col_index : int
        Index of the tenure column in the input array (default 0).
    """

    def __init__(self, threshold: int = 6, tenure_col_index: int = 0) -> None:
        self.threshold = threshold
        self.tenure_col_index = tenure_col_index

    def fit(self, X: pd.DataFrame, y: Any = None) -> "NewCustomerTransformer":
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        tenure = X.iloc[:, self.tenure_col_index].fillna(0)
        result = pd.DataFrame(
            {"is_new_customer": (tenure <= self.threshold).astype(float)},
            index=X.index,
        )
        return result

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(["is_new_customer"])


def _to_float(X: pd.DataFrame) -> pd.DataFrame:
    """Cast boolean columns to float for SimpleImputer compatibility."""
    return X.astype(float)


def load_feature_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load feature engineering configuration from YAML."""
    if config_path is None:
        config_path = Path("conf/data_preparation/features.yml")
    config_path = Path(config_path)
    with open(config_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def build_feature_pipeline(config: dict[str, Any] | None = None) -> ColumnTransformer:
    """Build an unfitted ColumnTransformer for churn feature engineering.

    Parameters
    ----------
    config : dict or None
        Feature configuration dict. If None, loads from default YAML path.

    Returns
    -------
    ColumnTransformer
        Unfitted transformer with sub-pipelines for each feature group.
    """
    if config is None:
        config = load_feature_config()

    numeric_impute = config.get("numeric_impute_strategy", "median")
    cat_impute = config.get("categorical_impute_strategy", "most_frequent")
    bool_impute = config.get("boolean_impute_strategy", "most_frequent")
    replacements = config.get("service_level_replacements", {})
    contract_order = config.get("contract_order", ["Month-to-month", "One year", "Two year"])
    threshold = config.get("new_customer_threshold", 6)

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=numeric_impute)),
        ]
    )

    boolean_pipeline = Pipeline(
        [
            ("to_float", FunctionTransformer(_to_float)),
            ("imputer", SimpleImputer(strategy=bool_impute)),
        ]
    )

    consolidate_ohe_pipeline = Pipeline(
        [
            ("consolidator", ServiceLevelConsolidator(replacements=replacements)),
            ("imputer", SimpleImputer(strategy=cat_impute)),
            ("encoder", OneHotEncoder(drop="if_binary", sparse_output=False)),
        ]
    )

    multi_ohe_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=cat_impute)),
            ("encoder", OneHotEncoder(drop="if_binary", sparse_output=False)),
        ]
    )

    contract_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=cat_impute)),
            ("encoder", OrdinalEncoder(categories=[contract_order])),
        ]
    )

    new_customer_pipeline = Pipeline(
        [
            ("new_customer", NewCustomerTransformer(threshold=threshold)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, config["numeric_columns"]),
            ("boolean", boolean_pipeline, config["boolean_columns"]),
            ("consolidate_ohe", consolidate_ohe_pipeline, config["consolidate_ohe_columns"]),
            ("multi_ohe", multi_ohe_pipeline, config["multi_ohe_columns"]),
            ("contract", contract_pipeline, [config["contract_column"]]),
            ("new_customer", new_customer_pipeline, config["numeric_columns"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")

    return preprocessor

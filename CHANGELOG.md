# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Churn Prediction Demo App**: Interactive Streamlit application with three tabs
  for individual customer prediction, batch CSV upload with filtered results, and
  model insights visualization (feature coefficients chart, threshold comparison).
- **Inference Module**: Reusable `src/inference/predict.py` with functions to load
  models, build customer dataframes, predict churn probability, and classify risk
  levels (Low / Medium / High).
- **Serving Configuration**: `conf/model_serving/churn.yml` defining model paths,
  input column schema, and risk level thresholds.
- **Threshold Optimization & Model Interpretation**: Evaluation module that finds
  the F1-optimal classification threshold (0.59) and extracts logistic regression
  coefficients for interpretability.
- **Hyperparameter Tuning**: GridSearchCV module for logistic regression with
  ElasticNet regularization (best C=0.5, l1_ratio=0.0, ROC-AUC 0.846).
- **Baseline Model Training**: Cross-validated comparison of Logistic Regression,
  Random Forest, and Gradient Boosting. LR selected as best model (ROC-AUC 0.846).
  Full training pipeline with preprocessor, scaler, and model.
- **Feature Engineering Pipeline**: sklearn ColumnTransformer pipeline producing 22
  features from 16 input columns. Includes custom transformers
  (`ServiceLevelConsolidator`, `NewCustomerTransformer`), imputation, encoding, and
  scaling.
- **Data Exploration & Analysis**: EDA notebooks covering univariate/bivariate
  analysis, correlation matrices, and variable analysis for the Telco Customer Churn
  dataset (7,043 customers, 21 columns).
- **Data Loading & Cleaning**: Notebooks and pipelines for raw CSV ingestion,
  type casting, duplicate removal, and parquet export through the layered data
  architecture (raw -> intermediate -> primary).
- **Unit Tests**: Comprehensive test suites for data transformation (14 tests),
  model training (20 tests), hyperparameter tuning (14 tests), model evaluation
  (18 tests), inference prediction (18 tests), data validation (14 tests), and
  inference pipeline (4 tests) -- 102 tests total.
- **FTI Pipeline Entry Points**: Feature pipeline, training pipeline, and
  inference pipeline orchestration scripts in `src/pipelines/`.
- **Data Validation Module**: Config-driven `src/data/validation.py` with schema
  checks for expected columns, null fractions, numeric ranges, and allowed
  categorical values. Validation rules defined in `conf/data_validation/churn.yml`.

### Fixed

- Corrected tenure-TotalCharges correlation value in analysis notebook.
- Re-executed notebooks with missing cell outputs.

### Dependencies

- Added `streamlit` and `plotly` as optional `app` dependency group.
- Added `numpy`, `pandas`, `scikit-learn`, `scipy`, `jupyter`, `seaborn`,
  `matplotlib` as core/dev dependencies.

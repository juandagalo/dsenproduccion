# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Data science project using the FTI (Feature/Training/Inference) Pipeline architecture for building ML systems. Built with UV package manager and follows a layered data engineering convention.

## Essential Commands

### Environment Setup
- `make install_env` - Install dependencies and pre-commit hooks (run after cloning)
- `make install_data_libs` - Install pandas, numpy, scikit-learn, jupyter, matplotlib, seaborn
- `uv add <package>` - Add new production dependency
- `uv add --group dev <package>` - Add development dependency

### Testing & Quality
- `make test` - Run pytest with coverage report
- `make test_verbose` - Run pytest with verbose output
- `make check` - Run all pre-commit hooks (ruff, mypy, commitizen)
- `uv run pytest tests/<module>` - Run specific test module
- `uv run pytest tests/<file>.py::<test_name>` - Run single test

### Documentation
- `make docs` - Serve documentation locally with MkDocs
- `make docs_test` - Build docs to check for errors

## Architecture

### FTI Pipeline Structure

**src/** contains three main pipeline types:
- **feature_pipeline**: Transforms raw data → features and labels
- **training_pipeline**: Transforms features and labels → trained model
- **inference_pipeline**: Takes features + trained model → predictions

**Source modules** (used by pipelines):
- **data/**: Data extraction, validation, processing, transformation
- **model/**: Model training, evaluation, validation, export
- **inference/**: Model prediction, serving, monitoring

### Data Layer Convention

Data flows through 8 layers in **data/** directory:
1. **01_raw**: Immutable source data (never modified)
2. **02_intermediate**: Typed data from raw sources
3. **03_primary**: Cleansed, domain-specific data
4. **04_feature**: Engineered features grouped by analysis area
5. **05_model_input**: All features for model training
6. **06_models**: Serialized trained models
7. **07_model_output**: Model predictions and results
8. **08_reporting**: Combined data for dashboards/reports

### Configuration

**conf/** directory contains pipeline configuration modules:
- config.yml - Main configuration file
- Individual modules: data_extraction, data_preparation, data_validation, model_train, model_evaluation, model_validation, model_serving

### Notebooks

**notebooks/** organized by workflow stage:
- 1-data: Data extraction and cleaning
- 2-exploration: EDA
- 3-analysis: Statistical analysis
- 4-feat_eng: Feature engineering
- 5-models: Model training and tuning
- 6-interpretation: Model interpretation
- 7-deploy: Deployment strategies
- 8-reports: Summaries and conclusions

## Code Quality Standards

### Ruff Configuration (.code_quality/ruff.toml)
- Line length: 100 characters
- Target: Python 3.10+
- Includes Jupyter notebooks (.ipynb)
- Enabled checks: bugbear, mccabe, pycodestyle, pyflakes, pylint, isort, bandit, pyupgrade, simplify, tryceratops
- Max complexity: 10

### MyPy Configuration (.code_quality/mypy.ini)
- Strict mode: disallow_untyped_defs, disallow_untyped_calls enabled
- All functions must have type annotations
- ignore_missing_imports: True

### Pre-commit Hooks
Auto-run before commits:
- YAML validation
- Large file detection (max 100MB)
- Private key detection
- Ruff linting and formatting
- MyPy type checking
- Commitizen (conventional commits)

## Development Workflow

1. Create/switch branches with descriptive names
2. Write code following FTI pipeline architecture
3. Add type hints for all functions (mypy enforces this)
4. Write tests in tests/ matching src/ structure
5. Pre-commit hooks run automatically on commit
6. Run `make check` to verify code quality manually
7. Run `make test` to ensure tests pass

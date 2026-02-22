# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Data science project using the FTI (Feature/Training/Inference) Pipeline architecture for building ML systems. Built with UV package manager (Python ≥3.12). Do not use pip directly — always use `uv`.

## Essential Commands

### Environment Setup
- `make install_env` - Install dependencies and pre-commit hooks (run after cloning)
- `make install_data_libs` - Install pandas, numpy, scikit-learn, jupyter, matplotlib, seaborn
- `uv add <package>` - Add new production dependency
- `uv add --group dev <package>` - Add development dependency
- `make clean_env` - Remove `.venv` virtual environment

### Testing & Quality
- `make test` - Run pytest with coverage report
- `make test_verbose` - Run pytest with verbose output
- `make test_coverage` - Generate `coverage.xml` for CI
- `make check` - Run all pre-commit hooks (ruff, mypy, commitizen)
- `make lint` - Run ruff only (faster than `make check`)
- `uv run pytest tests/<file>.py::<test_name>` - Run single test

### Git Helpers
- `make switch_main` - Switch to main and pull
- `make clean_branchs` - Delete local branches already removed from remote
- `make pre-commit_update` - Update pre-commit hook versions

### Documentation
- `make docs` - Serve documentation locally with MkDocs
- `make docs_test` - Build docs to check for errors

## Architecture

### FTI Pipeline Structure

**`src/pipelines/`** — executable pipeline entry points:
- `feature_pipeline/` — raw data → features & labels
- `training_pipeline/` — features & labels → trained model
- `inference_pipeline/` — features + model → predictions

**`src/`** — reusable modules consumed by pipelines:
- `data/` — extraction, validation, processing, transformation
- `model/` — training, evaluation, validation, export
- `inference/` — prediction, serving, monitoring

`pythonpath = ["src"]` is configured in `pyproject.toml`, so pipeline code can import `src` modules directly without package installation.

`src/tmp_mock.py` and `tests/test_mock.py` are scaffolding placeholders — delete them once real source files exist.

### Data Layer Convention

Data flows through numbered layers in `data/` — **never write back to an earlier layer**:

| Layer | Purpose |
|---|---|
| `01_raw` | Immutable source data |
| `02_intermediate` | Typed data from raw |
| `03_primary` | Cleansed domain data |
| `04_feature` | Engineered features |
| `05_model_input` | All features for training |
| `06_models` | Serialized models |
| `07_model_output` | Predictions |
| `08_reporting` | Dashboard/report data |

### Configuration

**`conf/`** contains YAML config modules per pipeline stage: `data_extraction`, `data_preparation`, `data_validation`, `model_train`, `model_evaluation`, `model_validation`, `model_serving`.

### Notebooks

Notebooks in `notebooks/` follow a numbered workflow stage:
`1-data` → `2-exploration` → `3-analysis` → `4-feat_eng` → `5-models` → `6-interpretation` → `7-deploy` → `8-reports`

**Naming convention:** `{consecutive}_{jdg}_{nameOfTheNotebook}_{YYYYMMDD}.ipynb`
- `consecutive`: zero-padded sequential number within the subfolder (e.g., `01`, `02`)
- `jdg`: fixed author initials, always literal
- `nameOfTheNotebook`: short `snake_case` description
- `YYYYMMDD`: creation date

Example: `01_jdg_exploratory_data_analysis_20260221.ipynb`

Use `notebooks/notebook_template.ipynb` as the starting point for new notebooks.

## Code Quality Standards

### Key Rules
- **All functions must have type annotations** — mypy enforces `disallow_untyped_defs` and `disallow_untyped_calls`.
- **Commits must follow Conventional Commits** — commitizen hook enforces this on every commit.
- **Tests mirror `src/` structure**: `src/data/loader.py` → `tests/data/test_loader.py`.
- **`assert` statements (S101) are allowed in tests** (bandit exception configured).

### Ruff (`.code_quality/ruff.toml`)
- Line length: 100, double quotes, Python 3.12+, max complexity: 10
- Applies to `.ipynb` notebooks as well
- Checks: bugbear, mccabe, pycodestyle, pyflakes, pylint, isort, bandit, pyupgrade, simplify, tryceratops

### MyPy (`.code_quality/mypy.ini`)
- Strict: `disallow_untyped_defs`, `disallow_untyped_calls`, `ignore_missing_imports: True`

### Pre-commit Hooks (auto-run before commits)
YAML validation, large file detection (max 100MB), private key detection, Ruff, MyPy, Commitizen.

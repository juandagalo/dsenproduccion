# Copilot Instructions

## Build, Test & Lint

```bash
# Environment
make install_env          # Install deps + pre-commit hooks
make install_data_libs    # Add pandas, numpy, scikit-learn, jupyter, matplotlib, seaborn

# Testing
make test                 # pytest with coverage
make test_verbose         # pytest verbose with coverage
uv run pytest tests/path/to/test_file.py::test_function_name  # single test

# Quality
make check                # all pre-commit hooks (ruff + mypy + commitizen)
make lint                 # ruff only

# Docs
make docs                 # serve locally with MkDocs
```

Dependencies: `uv add <pkg>` (prod), `uv add --group dev <pkg>` (dev). Package manager is **UV** — do not use pip directly.

## Architecture: FTI Pipeline

This project uses the **Feature / Training / Inference** pipeline pattern. Code lives in two layers:

**`src/pipelines/`** — executable pipeline entry points:
- `feature_pipeline/` — raw data → features & labels
- `training_pipeline/` — features & labels → trained model
- `inference_pipeline/` — features + model → predictions

**`src/`** — reusable modules consumed by pipelines:
- `data/` — extraction, validation, processing, transformation
- `model/` — training, evaluation, validation, export
- `inference/` — prediction, serving, monitoring

**`conf/`** — YAML config modules per pipeline stage (`data_extraction`, `data_preparation`, `data_validation`, `model_train`, `model_evaluation`, `model_validation`, `model_serving`).

## Data Layer Convention

Data in `data/` flows through numbered layers — never write to earlier layers:

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

## Key Conventions

- **All functions must have type annotations** — mypy strict mode enforces `disallow_untyped_defs` and `disallow_untyped_calls`.
- **Tests mirror `src/` structure**: a module at `src/data/loader.py` gets tests at `tests/data/test_loader.py`. `pythonpath = ["src"]` is set in `pyproject.toml`, so imports work without package installation.
- **Commits must follow Conventional Commits** — commitizen hook enforces this on every commit and push.
- **Ruff config** is at `.code_quality/ruff.toml` (not `pyproject.toml`). Line length: 100, double quotes, complexity ≤ 10. `assert` statements (S101) are allowed in tests.
- **Notebooks** in `notebooks/` follow a numbered workflow: `1-data` → `2-exploration` → `3-analysis` → `4-feat_eng` → `5-models` → `6-interpretation` → `7-deploy` → `8-reports`.
- `src/tmp_mock.py` and `tests/test_mock.py` are placeholder scaffolding — delete them once real source files are added.

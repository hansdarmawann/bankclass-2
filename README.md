# Bankclass - Credit Score Classifier

Local demo project for training and running a credit score classification model.
The project includes a TDSP-style AutoML notebook, saved MLflow model artifacts,
sample data, and a Streamlit app for CSV or manual prediction.

## Project Layout

- `app/` - Streamlit app source.
- `scripts/` - setup, validation, conversion, and app runner scripts.
- `notebooks/` - active notebook: `credit_score_automl_tdsp.ipynb`.
- `archive/notebooks/` - historical notebook versions.
- `data/` - small sample data and local runtime database.
- `train_silver/` and `test_silver/` - local parquet/Delta-style datasets.
- `mlruns/` - local MLflow model artifacts required for demo scoring.
- `outputs/` - generated logs, figures, and prediction exports.
- `requirements-notebook.txt` - notebook/training environment reference.

## Quick Start

Create the recommended conda environment:

```powershell
.\scripts\create_env.ps1
```

Or from Command Prompt:

```bat
scripts\create_env.bat
```

Run the setup check:

```powershell
python scripts\setup_check.py
```

Start the Streamlit app:

```powershell
.\scripts\run_streamlit.ps1
```

Or from Command Prompt:

```bat
scripts\run_streamlit.bat
```

Direct launch:

```powershell
python -m streamlit run app\streamlit_app.py
```

## Common Tasks

Validate the active notebook:

```powershell
python scripts\validate_notebook.py
```

Convert `test_silver/` parquet data to CSV:

```powershell
python scripts\convert_delta_to_csv.py
```

The CSV output is written to:

```text
test_silver/test_data.csv
```

## Notes

- Paths are derived from the repository root, so the project can be moved or cloned.
- The app uses the local MLflow artifact for `credit_score_classifier` when available.
- `data/credit_predictions.db` is a local runtime database and is ignored for future commits.
- Generated logs, figures, and predictions should live under `outputs/`.

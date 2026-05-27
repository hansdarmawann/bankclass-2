# Streamlit App

This app scores credit data with the local MLflow model and stores prediction
history in SQLite.

## Run

From the repository root:

```powershell
.\scripts\run_streamlit.ps1
```

Or:

```bat
scripts\run_streamlit.bat
```

Direct command:

```powershell
python -m streamlit run app\streamlit_app.py
```

## Features

- Upload CSV files for batch prediction.
- Use the manual form for single-record prediction.
- Save prediction history to `data/credit_predictions.db`.
- Display prediction confidence and class probabilities.
- Download prediction results as CSV.

## Expected Inputs

CSV files should use the same feature names as the training data. The app reads
the model signature from the MLflow artifact and fills missing required columns
with safe defaults before scoring.

## Local Artifacts

- Model artifacts: `mlruns/1/models/`
- Training sample for form defaults: `train_silver/`
- Runtime database: `data/credit_predictions.db`
- Generated outputs: `outputs/`

SQLite is suitable for local demos. Use a server database for concurrent or
production use.

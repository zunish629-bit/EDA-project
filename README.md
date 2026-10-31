# Breast Cancer EDA Dashboard

Interactive Streamlit dashboard for the included `Breast_Cancer.csv` dataset.

## Run locally

From this folder:

```powershell
$env:UV_CACHE_DIR = (Join-Path (Get-Location) '.uv-cache')
uv venv .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv run --python .venv\Scripts\python.exe -m streamlit run app.py
```

The sidebar supports CSV upload and filters for outcome, stage, age, follow-up time, race, marital status, and receptor status. The app includes overview, stage comparisons, clinical/demographic charts, a searchable filtered table, and CSV downloads.

Expected columns include `Age`, `Status`, `6th Stage`, `Survival Months`, `Tumor Size`, `Estrogen Status`, and `Progesterone Status`.

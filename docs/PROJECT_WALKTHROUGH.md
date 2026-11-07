# Project walkthrough

## What the project does

The project tests whether Ridge regression can improve monthly Niño 1+2 temperature forecasts over three simple baselines. It uses a historical series, creates lagged inputs, selects model regularisation using only earlier data, and evaluates all methods on a later holdout period.

## Main files

- `src/forecasting.py` contains data validation, feature creation, chronological splitting, Ridge alpha selection, baseline generation, scoring and the moving-block bootstrap.
- `src/run_forecast.py` runs the full experiment and refreshes the charts, metrics, predictions, validation scores and notes.
- `scripts/run_project.py` runs the forecast and the tests in one go, then saves a readable run log.
- `tests/test_forecasting.py` checks data handling, feature timing, baseline calculations, chronological splitting, model selection, scoring and bootstrap reproducibility.
- `data/elnino_monthly.csv` contains the supplied historical observations.

## Run the analysis

From the project root, with the virtual environment active:

```powershell
python scripts/run_project.py
```

The script writes generated outputs to `figures/` and `results/`.

## Understand the four methods

1. Persistence carries forward the previous month's observation.
2. Seasonal naive carries forward the observation from the same month one year earlier.
3. Training monthly mean uses the average for that calendar month, calculated only from the training period.
4. Ridge regression uses several past observations and calendar features. Its `alpha` is selected with time-ordered validation inside the training period.

## Forecast timing

The forecasts are one-step-ahead forecasts using observed history; no recursive prediction is used. Earlier observed test-month values may be used as lags for later test months. This is not the same as forecasting many months ahead from a single starting date.

## Review the generated files

- `figures/series_and_split.png` shows the complete historical series and the beginning of the test period.
- `figures/test_forecasts.png` compares observed test values with all four methods.
- `figures/model_mae_comparison.png` compares the four MAE scores.
- `results/metrics.csv` contains MAE and RMSE for each method.
- `results/test_predictions.csv` contains month-by-month forecasts.
- `results/alpha_validation.csv` records the validation score for each Ridge alpha candidate.
- `results/mae_difference_bootstrap.csv` contains paired 95% intervals for MAE differences.
- `results/EXPERIMENT_NOTES.md` summarises the latest execution.
- `logs/project_run.log` records the latest forecast and test run.

## Run the tests

Both commands are supported from the project root:

```powershell
pytest
python -m pytest
```

The `pytest.ini` file sets the repository root on the import path.

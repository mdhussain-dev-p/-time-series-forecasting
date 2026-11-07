# Start here — Time-Series Forecasting

The complete project and its dataset are included in this folder. MySQL is not required.

## Open the project

1. Extract the ZIP into a folder named `Time-Series-Forecasting`.
2. In VS Code, choose **File → Open Folder** and select that folder.
3. Open **Terminal → New Terminal** and check that the terminal is in the folder containing `src`, `data`, `tests` and `requirements.lock`.

## Set up Python on Windows

The recorded Windows environment for this project used Python 3.13.7. Check your version first:

```powershell
python --version
```

Create and activate the environment, then install the exact package versions from the lock file:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
```

## Run the project and tests

Run both commands from the repository root:

```powershell
python scripts/run_project.py
```

This single command runs the forecast, runs the automated tests, and saves the terminal output to `logs/project_run.log`. You can also run `python -m src.run_forecast` and `python -m pytest` separately. Plain `pytest` is supported too.

## See the output

- Open `figures/series_and_split.png` and `figures/test_forecasts.png` to see the historical series and test forecasts.
- Open `figures/model_mae_comparison.png` to compare the four methods by MAE.
- Open `results/metrics.csv` to compare all four forecasting methods.
- Open `results/test_predictions.csv` to inspect each test-month prediction.
- Open `results/alpha_validation.csv` to see how Ridge `alpha` was selected.
- Open `results/mae_difference_bootstrap.csv` to inspect the MAE-difference confidence intervals.
- Open `results/EXPERIMENT_NOTES.md` for the generated run summary.
- Open `logs/project_run.log` to review the latest forecast and test output.

For an explanation of the analysis, see [`docs/PROJECT_WALKTHROUGH.md`](docs/PROJECT_WALKTHROUGH.md) and [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). Dataset details are in [`data/DATA_SOURCE.md`](data/DATA_SOURCE.md).

If a command fails, copy the complete terminal error and review it before changing files or package versions.

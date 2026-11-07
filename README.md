# Monthly Sea-Surface Temperature Forecasting

## Project overview

This project compares a simple forecasting rule with a small machine-learning
model. The question is practical: **does Ridge regression make a more accurate
one-month-ahead forecast than simply carrying forward the previous month's
sea-surface temperature?**

The work follows a straightforward process: inspect the observations, create
features from earlier months, keep the dates in order, fit the model, compare
both methods on the same test period, and save the results.

## Dataset

The project includes `data/elnino_monthly.csv`, so it can run without a separate
download. It contains 732 monthly observations from January 1950 through
December 2010. The values are monthly average sea-surface temperatures in
degrees Celsius for the Niño 1+2 area of the Pacific Ocean.

The series is distributed with the `statsmodels` El Niño dataset and is
attributed there to NOAA's ERSST.V3B data. The statsmodels data page identifies
the dataset as public domain. See `data/DATA_SOURCE.md` for the source details
and links.

## Methods

### 1. Naive baseline

For each target month, the baseline uses the previous month's observed
temperature as its forecast. It is deliberately simple and gives the model a
clear point of comparison.

### 2. Ridge regression

The model uses temperatures from 1, 2, 3, 6 and 12 months earlier. It also uses
the calendar month, represented by sine and cosine values, to provide a simple
way to account for the annual cycle.

The model is a scikit-learn pipeline with two steps:

1. `StandardScaler` puts the input features on comparable scales.
2. `Ridge` regression fits a regularised linear model (`alpha=10.0`).

The scaler is fitted as part of the pipeline using only the training data.

## Evaluation design

The first 12 months are needed to build the longest lag. That leaves 720 usable
rows. The earliest 80% (576 months) are used for training, and the final 20%
(144 months) are used for testing. The dates are never shuffled.

The test period is January 1999 to December 2010. This is a **rolling
one-month-ahead evaluation**: the model is fitted once on the training period,
but once an actual test-month observation has arrived, it can be used as an
input for the next month's forecast. This is not a 12-year forecast made once
at the start of 1999.

The project reports two error measures:

- **MAE (Mean Absolute Error):** the average absolute difference between the
  forecast and the observed temperature.
- **RMSE (Root Mean Squared Error):** the square root of the average squared
  error, which gives larger mistakes more influence.

Both are measured in degrees Celsius. Lower scores indicate smaller errors.

## Results from the bundled run

| Method | MAE (°C) | RMSE (°C) |
|---|---:|---:|
| Naive baseline | 0.9942 | 1.1717 |
| Ridge regression | 0.4184 | 0.5344 |

On this particular test period, Ridge regression reduced MAE by approximately
57.9% compared with the naive baseline. These results are reproduced by the
project script and are not a guarantee that Ridge will outperform a baseline
on every dataset.

The project saves:

- `results/metrics.csv` — MAE and RMSE for each method.
- `results/test_predictions.csv` — observed values and both forecasts for each
  test month.
- `results/EXPERIMENT_NOTES.md` — a short report written from the latest run.
- `figures/series_and_split.png` — the complete series and the start of the
  test period.
- `figures/test_forecasts.png` — observed values and forecasts during testing.

## Run it on Windows

Python 3.10 or newer is recommended. The dataset is already included.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.run_forecast
python -m pytest -q
```

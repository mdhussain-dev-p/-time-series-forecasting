# Project walkthrough

This guide explains what the project does and where to find each part. The aim
is to make the work understandable, not just executable.

## Start with the main question

A time-series model should be compared with a sensible simple forecast. Here,
the simple forecast says that the next month's temperature will be the same as
the previous month's observed temperature. Ridge regression gets a little more
context by looking at several earlier months and the month of the year.

## `src/forecasting.py`

This file contains the reusable parts of the analysis:

- `load_monthly_series()` reads the CSV and checks its dates and values.
- `make_feature_table()` creates the lagged inputs and calendar features.
- `split_by_time()` keeps the earlier dates for training and later dates for
  testing.
- `build_forecasting_model()` creates the scaling and Ridge regression
  pipeline.
- `score_forecast()` calculates MAE and RMSE after checking that the values
  being compared line up.

Keeping these steps in functions makes them easier to test and easier to read.

## `src/run_forecast.py`

This is the entry point for the full experiment. It loads the observations,
creates features, trains the model, generates test forecasts, calculates the
scores, and saves the charts and result files.

Run it from the project folder with:

```powershell
python -m src.run_forecast
```

## `tests/test_forecasting.py`

The tests focus on mistakes that could quietly make a result unreliable. They
check the date range, duplicate and missing months, missing values, feature
timing, chronological splitting, metric calculations and whether the model
can fit and return finite forecasts.

Run them with:

```powershell
python -m pytest -q
```

## What is a lag?

A lag is an earlier observation. For a forecast for January 2001, `lag_1` is
the observed value from December 2000 and `lag_12` is the observed value from
January 2000. The tests check this timing so the model does not accidentally
receive the value it is supposed to predict.

## Why use a time-based split?

A random split could put some later months in the training set and earlier
months in the test set. That would not match the forecasting situation being
studied. This project trains on the earlier dates and tests on the later dates.

## What do the error measures mean?

- **MAE** is the average absolute forecast error. It is easy to interpret in
  the original temperature units.
- **RMSE** also uses the original units, but larger errors have more influence
  because the errors are squared before averaging.

For both measures, lower is better. Neither score, on its own, proves that a
model will work well on another dataset or time period.

## Questions to check your understanding

1. Why is the naive baseline included?
2. Why are the observations kept in date order?
3. What does `lag_12` mean for a forecast made in January?
4. Is the Ridge model re-fitted during the test period?
5. Why is the test result specific to this dataset and forecast setup?

You should be able to answer these in your own words before presenting the
project to someone else.

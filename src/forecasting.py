from __future__ import annotations

from numbers import Real
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LAG_MONTHS = (1, 2, 3, 6, 12)
MODEL_FEATURES = tuple(f"lag_{lag}" for lag in LAG_MONTHS) + (
    "month_sin",
    "month_cos",
)
ALPHA_CANDIDATES = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
VALIDATION_SPLITS = 5
VALIDATION_MONTHS = 60
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_BLOCK_MONTHS = 12
BOOTSTRAP_SEED = 2026


def load_monthly_series(csv_path: str | Path) -> pd.Series:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = pd.read_csv(path)
    required_columns = {"date", "sst_c"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required column(s): {missing_text}")
    if data.empty:
        raise ValueError("The dataset is empty.")

    data = data.loc[:, ["date", "sst_c"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["sst_c"] = pd.to_numeric(data["sst_c"], errors="coerce")

    if data["date"].isna().any():
        raise ValueError("One or more dates could not be read.")
    if data["sst_c"].isna().any():
        raise ValueError("One or more temperature values are missing or invalid.")
    if not np.isfinite(data["sst_c"].to_numpy(dtype=float)).all():
        raise ValueError("Temperature values must be finite numbers.")

    data = data.sort_values("date")
    if data["date"].duplicated().any():
        raise ValueError("The dataset contains duplicate dates.")

    series = data.set_index("date")["sst_c"].astype(float)
    expected_dates = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq="MS",
    )
    if not series.index.equals(expected_dates):
        raise ValueError(
            "The dataset must contain one observation for every month, "
            "with dates on the first day of each month."
        )

    series.name = "sst_c"
    series.index.name = "date"
    return series


def make_feature_table(
    series: pd.Series,
    lags: Iterable[int] = LAG_MONTHS,
) -> pd.DataFrame:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("The series must have a DatetimeIndex.")
    if not series.index.is_monotonic_increasing or series.index.has_duplicates:
        raise ValueError("The series dates must be sorted and unique.")
    if series.empty:
        raise ValueError("The series is empty.")

    expected_dates = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq="MS",
    )
    if not series.index.equals(expected_dates):
        raise ValueError("The series must contain consecutive month-start dates.")

    values = pd.to_numeric(series, errors="coerce").astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Temperature values must be finite numbers.")

    lag_values = tuple(lags)
    if not lag_values:
        raise ValueError("Provide at least one lag.")
    if any(
        not isinstance(lag, (int, np.integer))
        or isinstance(lag, (bool, np.bool_))
        or lag < 1
        for lag in lag_values
    ):
        raise ValueError("Every lag must be a positive whole number.")
    if len(set(lag_values)) != len(lag_values):
        raise ValueError("Do not provide the same lag more than once.")

    table = pd.DataFrame({"actual_sst_c": values})
    for lag in lag_values:
        table[f"lag_{lag}"] = values.shift(lag)

    month_angle = 2 * np.pi * table.index.month / 12
    table["month_sin"] = np.sin(month_angle)
    table["month_cos"] = np.cos(month_angle)
    table["persistence_prediction_c"] = values.shift(1)
    table["seasonal_naive_prediction_c"] = values.shift(12)
    return table.dropna().copy()


def split_by_time(
    table: pd.DataFrame,
    train_fraction: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if (
        isinstance(train_fraction, (bool, np.bool_))
        or not isinstance(train_fraction, Real)
        or not np.isfinite(train_fraction)
        or not 0.5 <= train_fraction < 1.0
    ):
        raise ValueError("train_fraction must be from 0.5 up to, but not including, 1.0.")
    if len(table) < 2:
        raise ValueError("At least two rows are needed for a train/test split.")
    if not isinstance(table.index, pd.DatetimeIndex):
        raise ValueError("The table must have a DatetimeIndex.")
    if not table.index.is_monotonic_increasing or table.index.has_duplicates:
        raise ValueError("The table dates must be sorted and unique.")

    split_position = int(len(table) * train_fraction)
    training_data = table.iloc[:split_position].copy()
    test_data = table.iloc[split_position:].copy()

    if training_data.empty or test_data.empty:
        raise ValueError("The split produced an empty training or test set.")
    return training_data, test_data


def build_forecasting_model(alpha: float) -> Pipeline:
    if (
        not isinstance(alpha, Real)
        or isinstance(alpha, (bool, np.bool_))
        or not np.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("alpha must be a positive number.")
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


def select_ridge_alpha(
    training_data: pd.DataFrame,
    candidates: Iterable[float] = ALPHA_CANDIDATES,
    n_splits: int = VALIDATION_SPLITS,
    validation_months: int = VALIDATION_MONTHS,
) -> tuple[float, pd.DataFrame]:
    candidate_values = tuple(float(value) for value in candidates)
    if not candidate_values or any(not np.isfinite(value) or value <= 0 for value in candidate_values):
        raise ValueError("Alpha candidates must be positive finite numbers.")
    if len(set(candidate_values)) != len(candidate_values):
        raise ValueError("Alpha candidates must be unique.")
    if n_splits < 2 or validation_months < 1:
        raise ValueError("Time-ordered validation needs at least two splits and one month per split.")
    if len(training_data) <= n_splits * validation_months:
        raise ValueError("Not enough training observations for the requested validation windows.")

    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=validation_months)
    rows = []

    for alpha in candidate_values:
        fold_scores = []
        for train_positions, validation_positions in splitter.split(training_data):
            fold_train = training_data.iloc[train_positions]
            fold_validation = training_data.iloc[validation_positions]
            model = build_forecasting_model(alpha)
            model.fit(fold_train.loc[:, MODEL_FEATURES], fold_train["actual_sst_c"])
            forecast = model.predict(fold_validation.loc[:, MODEL_FEATURES])
            fold_scores.append(
                mean_absolute_error(fold_validation["actual_sst_c"], forecast)
            )
        rows.append(
            {
                "alpha": alpha,
                "mean_validation_mae_c": float(np.mean(fold_scores)),
                "std_validation_mae_c": float(np.std(fold_scores, ddof=1)),
            }
        )

    selection = pd.DataFrame(rows).sort_values(
        ["mean_validation_mae_c", "alpha"],
        ascending=[True, True],
    ).reset_index(drop=True)
    selected_alpha = float(selection.loc[0, "alpha"])
    selection["selected"] = selection["alpha"].eq(selected_alpha)
    return selected_alpha, selection.sort_values("alpha").reset_index(drop=True)


def monthly_mean_forecast(
    training_values: pd.Series,
    forecast_dates: pd.DatetimeIndex,
) -> pd.Series:
    if not isinstance(training_values.index, pd.DatetimeIndex):
        raise ValueError("Training values must have a DatetimeIndex.")
    if not isinstance(forecast_dates, pd.DatetimeIndex):
        raise ValueError("Forecast dates must be a DatetimeIndex.")
    if training_values.empty or forecast_dates.empty:
        raise ValueError("Training values and forecast dates cannot be empty.")

    monthly_means = training_values.groupby(training_values.index.month).mean()
    predictions = pd.Series(
        forecast_dates.month.map(monthly_means),
        index=forecast_dates,
        name="training_monthly_mean_prediction_c",
        dtype=float,
    )
    if predictions.isna().any():
        raise ValueError("The training data does not contain an average for every forecast month.")
    return predictions


def score_forecast(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict[str, float]:
    if len(actual) != len(predicted):
        raise ValueError("Actual values and predictions must have equal lengths.")
    if len(actual) == 0:
        raise ValueError("There are no predictions to score.")
    if isinstance(actual, pd.Series) and isinstance(predicted, pd.Series):
        if not actual.index.equals(predicted.index):
            raise ValueError("Actual values and predictions must use the same dates.")

    try:
        actual_values = np.asarray(actual, dtype=float)
        predicted_values = np.asarray(predicted, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Actual values and predictions must be numeric.") from error

    if actual_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("Actual values and predictions must be one-dimensional.")
    if not np.isfinite(actual_values).all():
        raise ValueError("Actual values contain a missing or infinite number.")
    if not np.isfinite(predicted_values).all():
        raise ValueError("Predictions contain a missing or infinite number.")

    return {
        "MAE_C": float(mean_absolute_error(actual_values, predicted_values)),
        "RMSE_C": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
    }


def moving_block_bootstrap_mae_difference(
    actual: pd.Series,
    first_forecast: pd.Series,
    second_forecast: pd.Series,
    replicates: int = BOOTSTRAP_REPLICATES,
    block_length: int = BOOTSTRAP_BLOCK_MONTHS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    if not actual.index.equals(first_forecast.index) or not actual.index.equals(second_forecast.index):
        raise ValueError("All series must use the same dates.")
    if len(actual) == 0:
        raise ValueError("At least one test observation is required.")
    if replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")
    if block_length < 1 or block_length > len(actual):
        raise ValueError("block_length must be between 1 and the number of observations.")

    actual_values = actual.to_numpy(dtype=float)
    first_errors = np.abs(actual_values - first_forecast.to_numpy(dtype=float))
    second_errors = np.abs(actual_values - second_forecast.to_numpy(dtype=float))
    differences = first_errors - second_errors
    if not np.isfinite(differences).all():
        raise ValueError("Forecast errors must be finite.")

    rng = np.random.default_rng(seed)
    sample_count = len(differences)
    blocks_needed = int(np.ceil(sample_count / block_length))
    offsets = np.arange(block_length)
    estimates = np.empty(replicates, dtype=float)

    for replicate in range(replicates):
        starts = rng.integers(0, sample_count, size=blocks_needed)
        sample_indices = ((starts[:, None] + offsets) % sample_count).ravel()[:sample_count]
        estimates[replicate] = differences[sample_indices].mean()

    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "mae_difference_c": float(differences.mean()),
        "ci_95_lower_c": float(lower),
        "ci_95_upper_c": float(upper),
        "bootstrap_replicates": int(replicates),
        "block_length_months": int(block_length),
    }

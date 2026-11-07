from __future__ import annotations
from numbers import Real
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
LAG_MONTHS = (1, 2, 3, 6, 12)
MODEL_FEATURES = tuple((f'lag_{lag}' for lag in LAG_MONTHS)) + ('month_sin', 'month_cos')

def load_monthly_series(csv_path: str | Path) -> pd.Series:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f'Dataset file not found: {path}')
    data = pd.read_csv(path)
    required_columns = {'date', 'sst_c'}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing_text = ', '.join(sorted(missing_columns))
        raise ValueError(f'Dataset is missing required column(s): {missing_text}')
    if data.empty:
        raise ValueError('The dataset is empty.')
    data = data.loc[:, ['date', 'sst_c']].copy()
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['sst_c'] = pd.to_numeric(data['sst_c'], errors='coerce')
    if data['date'].isna().any():
        raise ValueError('One or more dates could not be read.')
    if data['sst_c'].isna().any():
        raise ValueError('One or more temperature values are missing or invalid.')
    if not np.isfinite(data['sst_c'].to_numpy(dtype=float)).all():
        raise ValueError('Temperature values must be finite numbers.')
    data = data.sort_values('date')
    if data['date'].duplicated().any():
        raise ValueError('The dataset contains duplicate dates.')
    series = data.set_index('date')['sst_c'].astype(float)
    expected_dates = pd.date_range(start=series.index.min(), end=series.index.max(), freq='MS')
    if not series.index.equals(expected_dates):
        raise ValueError('The dataset must contain one observation for every month, with dates on the first day of each month.')
    series.name = 'sst_c'
    series.index.name = 'date'
    return series

def make_feature_table(series: pd.Series, lags: Iterable[int]=LAG_MONTHS) -> pd.DataFrame:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError('The series must have a DatetimeIndex.')
    if not series.index.is_monotonic_increasing or series.index.has_duplicates:
        raise ValueError('The series dates must be sorted and unique.')
    if series.empty:
        raise ValueError('The series is empty.')
    expected_dates = pd.date_range(start=series.index.min(), end=series.index.max(), freq='MS')
    if not series.index.equals(expected_dates):
        raise ValueError('The series must contain consecutive month-start dates.')
    values = pd.to_numeric(series, errors='coerce').astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError('Temperature values must be finite numbers.')
    lag_values = tuple(lags)
    if not lag_values:
        raise ValueError('Provide at least one lag.')
    if any((not isinstance(lag, (int, np.integer)) or isinstance(lag, (bool, np.bool_)) or lag < 1 for lag in lag_values)):
        raise ValueError('Every lag must be a positive whole number.')
    if len(set(lag_values)) != len(lag_values):
        raise ValueError('Do not provide the same lag more than once.')
    table = pd.DataFrame({'actual_sst_c': values})
    for lag in lag_values:
        table[f'lag_{lag}'] = values.shift(lag)
    month_angle = 2 * np.pi * table.index.month / 12
    table['month_sin'] = np.sin(month_angle)
    table['month_cos'] = np.cos(month_angle)
    table['naive_prediction_c'] = values.shift(1)
    table = table.dropna().copy()
    if table.empty:
        raise ValueError('There are not enough observations to create the lagged features.')
    return table

def split_by_time(table: pd.DataFrame, train_fraction: float=0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(train_fraction, (bool, np.bool_)) or not isinstance(train_fraction, Real) or (not np.isfinite(train_fraction)) or (not 0.5 <= train_fraction < 1.0):
        raise ValueError('train_fraction must be a number from 0.5 up to, but not including, 1.0.')
    if len(table) < 2:
        raise ValueError('At least two rows are needed for a train/test split.')
    if not isinstance(table.index, pd.DatetimeIndex):
        raise ValueError('The table must have a DatetimeIndex.')
    if not table.index.is_monotonic_increasing or table.index.has_duplicates:
        raise ValueError('The table dates must be sorted and unique.')
    split_position = int(len(table) * train_fraction)
    training_data = table.iloc[:split_position].copy()
    test_data = table.iloc[split_position:].copy()
    if training_data.empty or test_data.empty:
        raise ValueError('The split produced an empty training or test set.')
    return (training_data, test_data)

def build_forecasting_model() -> Pipeline:
    return Pipeline(steps=[('scale', StandardScaler()), ('ridge', Ridge(alpha=10.0))])

def score_forecast(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    if len(actual) != len(predicted):
        raise ValueError('Actual values and predictions must have equal lengths.')
    if len(actual) == 0:
        raise ValueError('There are no predictions to score.')
    if isinstance(actual, pd.Series) and isinstance(predicted, pd.Series):
        if not actual.index.equals(predicted.index):
            raise ValueError('Actual values and predictions must use the same dates.')
    try:
        actual_values = np.asarray(actual, dtype=float)
        predicted_values = np.asarray(predicted, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('Actual values and predictions must be numeric.') from error
    if actual_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError('Actual values and predictions must be one-dimensional.')
    if not np.isfinite(actual_values).all():
        raise ValueError('Actual values contain a missing or infinite number.')
    if not np.isfinite(predicted_values).all():
        raise ValueError('Predictions contain a missing or infinite number.')
    return {'MAE_C': float(mean_absolute_error(actual_values, predicted_values)), 'RMSE_C': float(np.sqrt(mean_squared_error(actual_values, predicted_values)))}

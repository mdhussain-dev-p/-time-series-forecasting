from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.forecasting import MODEL_FEATURES, build_forecasting_model, load_monthly_series, make_feature_table, score_forecast, split_by_time
PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_FOLDER / 'data' / 'elnino_monthly.csv'

def test_bundled_dataset_has_the_expected_date_range_and_row_count():
    series = load_monthly_series(DATA_FILE)
    assert len(series) == 732
    assert series.index[0] == pd.Timestamp('1950-01-01')
    assert series.index[-1] == pd.Timestamp('2010-12-01')
    assert series.index.is_monotonic_increasing
    assert series.notna().all()

def test_loader_rejects_an_empty_dataset(tmp_path):
    csv_file = tmp_path / 'empty.csv'
    pd.DataFrame(columns=['date', 'sst_c']).to_csv(csv_file, index=False)
    with pytest.raises(ValueError, match='empty'):
        load_monthly_series(csv_file)

def test_loader_rejects_duplicate_dates(tmp_path):
    csv_file = tmp_path / 'duplicate_dates.csv'
    pd.DataFrame({'date': ['2000-01-01', '2000-01-01'], 'sst_c': [20.0, 21.0]}).to_csv(csv_file, index=False)
    with pytest.raises(ValueError, match='duplicate dates'):
        load_monthly_series(csv_file)

def test_loader_rejects_a_missing_month(tmp_path):
    csv_file = tmp_path / 'missing_month.csv'
    pd.DataFrame({'date': ['2000-01-01', '2000-03-01'], 'sst_c': [20.0, 21.0]}).to_csv(csv_file, index=False)
    with pytest.raises(ValueError, match='every month'):
        load_monthly_series(csv_file)

def test_loader_rejects_missing_temperature_values(tmp_path):
    csv_file = tmp_path / 'missing_value.csv'
    pd.DataFrame({'date': ['2000-01-01', '2000-02-01'], 'sst_c': [20.0, None]}).to_csv(csv_file, index=False)
    with pytest.raises(ValueError, match='missing or invalid'):
        load_monthly_series(csv_file)

def test_loader_rejects_infinite_temperature_values(tmp_path):
    csv_file = tmp_path / 'infinite_value.csv'
    pd.DataFrame({'date': ['2000-01-01', '2000-02-01'], 'sst_c': [20.0, np.inf]}).to_csv(csv_file, index=False)
    with pytest.raises(ValueError, match='finite numbers'):
        load_monthly_series(csv_file)

def test_first_feature_row_uses_only_the_previous_twelve_months():
    dates = pd.date_range('2000-01-01', periods=15, freq='MS')
    values = pd.Series(np.arange(15, dtype=float), index=dates)
    features = make_feature_table(values)
    first_forecast = features.iloc[0]
    assert first_forecast.name == pd.Timestamp('2001-01-01')
    assert first_forecast['actual_sst_c'] == 12.0
    assert first_forecast['lag_1'] == 11.0
    assert first_forecast['lag_12'] == 0.0
    assert first_forecast['naive_prediction_c'] == 11.0

def test_feature_builder_rejects_non_monthly_dates():
    dates = pd.to_datetime(['2000-01-01', '2000-02-01', '2000-04-01'])
    values = pd.Series([1.0, 2.0, 3.0], index=dates)
    with pytest.raises(ValueError, match='consecutive month-start'):
        make_feature_table(values)

def test_feature_builder_rejects_infinite_values():
    dates = pd.date_range('2000-01-01', periods=15, freq='MS')
    values = pd.Series(np.arange(15, dtype=float), index=dates)
    values.iloc[-1] = np.inf
    with pytest.raises(ValueError, match='finite numbers'):
        make_feature_table(values)

def test_feature_builder_rejects_non_positive_lags():
    dates = pd.date_range('2000-01-01', periods=15, freq='MS')
    values = pd.Series(np.arange(15, dtype=float), index=dates)
    with pytest.raises(ValueError, match='positive whole number'):
        make_feature_table(values, lags=(1, 0))

def test_feature_builder_rejects_duplicate_lags():
    dates = pd.date_range('2000-01-01', periods=15, freq='MS')
    values = pd.Series(np.arange(15, dtype=float), index=dates)
    with pytest.raises(ValueError, match='same lag'):
        make_feature_table(values, lags=(1, 1))

def test_time_split_keeps_the_latest_months_for_testing():
    series = load_monthly_series(DATA_FILE)
    features = make_feature_table(series)
    training_data, test_data = split_by_time(features)
    assert training_data.index.max() < test_data.index.min()
    assert test_data.index[0] == pd.Timestamp('1999-01-01')
    assert len(training_data) == 576
    assert len(test_data) == 144

def test_time_split_rejects_unsorted_dates():
    dates = pd.to_datetime(['2000-02-01', '2000-01-01', '2000-03-01'])
    table = pd.DataFrame({'value': [2, 1, 3]}, index=dates)
    with pytest.raises(ValueError, match='sorted and unique'):
        split_by_time(table)

def test_time_split_rejects_duplicate_dates():
    dates = pd.to_datetime(['2000-01-01', '2000-01-01', '2000-03-01'])
    table = pd.DataFrame({'value': [1, 2, 3]}, index=dates)
    with pytest.raises(ValueError, match='sorted and unique'):
        split_by_time(table)

@pytest.mark.parametrize('fraction', [1.0, 0.49, float('nan'), float('inf'), True])
def test_time_split_rejects_invalid_training_fractions(fraction):
    dates = pd.date_range('2000-01-01', periods=10, freq='MS')
    table = pd.DataFrame({'value': range(10)}, index=dates)
    with pytest.raises(ValueError, match='train_fraction'):
        split_by_time(table, train_fraction=fraction)

def test_metrics_match_a_small_hand_calculated_example():
    actual = pd.Series([2.0, 4.0, 6.0])
    predicted = pd.Series([1.0, 5.0, 6.0])
    scores = score_forecast(actual, predicted)
    assert scores['MAE_C'] == pytest.approx(2 / 3)
    assert scores['RMSE_C'] == pytest.approx(np.sqrt(2 / 3))

def test_metrics_reject_series_with_different_dates():
    actual = pd.Series([1.0, 2.0], index=pd.date_range('2000-01-01', periods=2, freq='MS'))
    predicted = pd.Series([1.0, 2.0], index=pd.date_range('2000-02-01', periods=2, freq='MS'))
    with pytest.raises(ValueError, match='same dates'):
        score_forecast(actual, predicted)

def test_metrics_reject_missing_or_infinite_values():
    actual = pd.Series([1.0, np.nan])
    predicted = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError, match='missing or infinite'):
        score_forecast(actual, predicted)
    with pytest.raises(ValueError, match='missing or infinite'):
        score_forecast(pd.Series([1.0, 2.0]), pd.Series([1.0, np.inf]))

def test_metrics_reject_non_numeric_values():
    with pytest.raises(ValueError, match='must be numeric'):
        score_forecast(['warm', 'cool'], [1.0, 2.0])

def test_ridge_pipeline_fits_and_returns_finite_predictions():
    series = load_monthly_series(DATA_FILE)
    features = make_feature_table(series)
    training_data, test_data = split_by_time(features)
    model = build_forecasting_model()
    model.fit(training_data.loc[:, MODEL_FEATURES], training_data['actual_sst_c'])
    predictions = model.predict(test_data.loc[:, MODEL_FEATURES])
    assert len(predictions) == len(test_data)
    assert np.isfinite(predictions).all()

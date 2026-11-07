from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forecasting import (
    ALPHA_CANDIDATES,
    MODEL_FEATURES,
    build_forecasting_model,
    load_monthly_series,
    make_feature_table,
    monthly_mean_forecast,
    moving_block_bootstrap_mae_difference,
    score_forecast,
    select_ridge_alpha,
    split_by_time,
)


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_FOLDER / "data" / "elnino_monthly.csv"


def test_bundled_dataset_has_expected_dates_and_source_values():
    series = load_monthly_series(DATA_FILE)

    assert len(series) == 732
    assert series.index[0] == pd.Timestamp("1950-01-01")
    assert series.index[-1] == pd.Timestamp("2010-12-01")
    assert series.loc["1950-01-01"] == pytest.approx(23.11)
    assert series.loc["1950-02-01"] == pytest.approx(24.20)
    assert series.loc["2010-12-01"] == pytest.approx(22.07)


def test_loader_rejects_an_empty_dataset(tmp_path):
    csv_file = tmp_path / "empty.csv"
    pd.DataFrame(columns=["date", "sst_c"]).to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="empty"):
        load_monthly_series(csv_file)


def test_loader_rejects_duplicate_dates(tmp_path):
    csv_file = tmp_path / "duplicate_dates.csv"
    pd.DataFrame(
        {"date": ["2000-01-01", "2000-01-01"], "sst_c": [20.0, 21.0]}
    ).to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="duplicate dates"):
        load_monthly_series(csv_file)


def test_loader_rejects_a_missing_month(tmp_path):
    csv_file = tmp_path / "missing_month.csv"
    pd.DataFrame(
        {"date": ["2000-01-01", "2000-03-01"], "sst_c": [20.0, 21.0]}
    ).to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="every month"):
        load_monthly_series(csv_file)


def test_loader_rejects_missing_temperature_values(tmp_path):
    csv_file = tmp_path / "missing_value.csv"
    pd.DataFrame(
        {"date": ["2000-01-01", "2000-02-01"], "sst_c": [20.0, None]}
    ).to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="missing or invalid"):
        load_monthly_series(csv_file)


def test_loader_rejects_infinite_temperature_values(tmp_path):
    csv_file = tmp_path / "infinite_value.csv"
    pd.DataFrame(
        {"date": ["2000-01-01", "2000-02-01"], "sst_c": [20.0, np.inf]}
    ).to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="finite numbers"):
        load_monthly_series(csv_file)


def test_feature_table_uses_only_prior_observations():
    dates = pd.date_range("2000-01-01", periods=15, freq="MS")
    values = pd.Series(np.arange(15, dtype=float), index=dates)
    features = make_feature_table(values)
    first_forecast = features.iloc[0]

    assert first_forecast.name == pd.Timestamp("2001-01-01")
    assert first_forecast["actual_sst_c"] == 12.0
    assert first_forecast["lag_1"] == 11.0
    assert first_forecast["lag_12"] == 0.0
    assert first_forecast["persistence_prediction_c"] == 11.0
    assert first_forecast["seasonal_naive_prediction_c"] == 0.0


def test_feature_builder_rejects_non_monthly_dates():
    dates = pd.to_datetime(["2000-01-01", "2000-02-01", "2000-04-01"])
    values = pd.Series([1.0, 2.0, 3.0], index=dates)

    with pytest.raises(ValueError, match="consecutive month-start"):
        make_feature_table(values)


def test_feature_builder_rejects_infinite_values():
    dates = pd.date_range("2000-01-01", periods=15, freq="MS")
    values = pd.Series(np.arange(15, dtype=float), index=dates)
    values.iloc[-1] = np.inf

    with pytest.raises(ValueError, match="finite numbers"):
        make_feature_table(values)


def test_feature_builder_rejects_invalid_and_duplicate_lags():
    dates = pd.date_range("2000-01-01", periods=15, freq="MS")
    values = pd.Series(np.arange(15, dtype=float), index=dates)

    with pytest.raises(ValueError, match="positive whole number"):
        make_feature_table(values, lags=(1, 0))
    with pytest.raises(ValueError, match="same lag"):
        make_feature_table(values, lags=(1, 1))


def test_time_split_keeps_latest_months_for_testing():
    features = make_feature_table(load_monthly_series(DATA_FILE))
    training_data, test_data = split_by_time(features)

    assert training_data.index.max() < test_data.index.min()
    assert test_data.index[0] == pd.Timestamp("1999-01-01")
    assert len(training_data) == 576
    assert len(test_data) == 144


def test_time_split_rejects_unsorted_and_duplicate_dates():
    unsorted_dates = pd.to_datetime(["2000-02-01", "2000-01-01", "2000-03-01"])
    duplicate_dates = pd.to_datetime(["2000-01-01", "2000-01-01", "2000-03-01"])

    with pytest.raises(ValueError, match="sorted and unique"):
        split_by_time(pd.DataFrame({"value": [2, 1, 3]}, index=unsorted_dates))
    with pytest.raises(ValueError, match="sorted and unique"):
        split_by_time(pd.DataFrame({"value": [1, 2, 3]}, index=duplicate_dates))


def test_time_split_rejects_invalid_training_fraction():
    dates = pd.date_range("2000-01-01", periods=10, freq="MS")
    table = pd.DataFrame({"value": range(10)}, index=dates)

    for fraction in (1.0, 0.49, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="train_fraction"):
            split_by_time(table, train_fraction=fraction)


def test_monthly_mean_baseline_uses_training_values_by_calendar_month():
    training_dates = pd.date_range("1990-01-01", "1991-12-01", freq="MS")
    training_values = pd.Series(
        [float(date.month) + (0.0 if date.year == 1990 else 2.0) for date in training_dates],
        index=training_dates,
    )
    forecast_dates = pd.to_datetime(["1992-01-01", "1992-06-01", "1992-12-01"])

    predictions = monthly_mean_forecast(training_values, forecast_dates)

    assert predictions.tolist() == pytest.approx([2.0, 7.0, 13.0])
    assert predictions.index.equals(forecast_dates)


def test_alpha_selection_uses_allowed_values_and_time_ordered_validation():
    training_data, _ = split_by_time(make_feature_table(load_monthly_series(DATA_FILE)))
    selected_alpha, selection = select_ridge_alpha(training_data)

    assert selected_alpha in ALPHA_CANDIDATES
    assert selection["selected"].sum() == 1
    assert selection["mean_validation_mae_c"].notna().all()
    assert (selection["mean_validation_mae_c"] > 0).all()


def test_ridge_model_rejects_invalid_alpha():
    for alpha in (0, -1, float("inf"), True):
        with pytest.raises(ValueError, match="positive number"):
            build_forecasting_model(alpha)


def test_metrics_match_hand_calculated_example():
    actual = pd.Series([2.0, 4.0, 6.0])
    predicted = pd.Series([1.0, 5.0, 6.0])
    scores = score_forecast(actual, predicted)

    assert scores["MAE_C"] == pytest.approx(2 / 3)
    assert scores["RMSE_C"] == pytest.approx(np.sqrt(2 / 3))


def test_metrics_reject_series_with_different_dates():
    actual = pd.Series([1.0, 2.0], index=pd.date_range("2000-01-01", periods=2, freq="MS"))
    predicted = pd.Series([1.0, 2.0], index=pd.date_range("2000-02-01", periods=2, freq="MS"))

    with pytest.raises(ValueError, match="same dates"):
        score_forecast(actual, predicted)


def test_metrics_reject_non_finite_and_non_numeric_values():
    with pytest.raises(ValueError, match="missing or infinite"):
        score_forecast(pd.Series([1.0, np.nan]), pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError, match="missing or infinite"):
        score_forecast(pd.Series([1.0, 2.0]), pd.Series([1.0, np.inf]))
    with pytest.raises(ValueError, match="must be numeric"):
        score_forecast(["warm", "cool"], [1.0, 2.0])


def test_moving_block_bootstrap_is_reproducible_and_returns_interval():
    dates = pd.date_range("2000-01-01", periods=36, freq="MS")
    actual = pd.Series(np.linspace(10, 20, 36), index=dates)
    first = actual + 0.5
    second = actual + 0.8

    first_result = moving_block_bootstrap_mae_difference(
        actual, first, second, replicates=300, block_length=12, seed=7
    )
    second_result = moving_block_bootstrap_mae_difference(
        actual, first, second, replicates=300, block_length=12, seed=7
    )

    assert first_result == second_result
    assert first_result["mae_difference_c"] == pytest.approx(-0.3)
    assert first_result["ci_95_lower_c"] <= first_result["ci_95_upper_c"]


def test_ridge_pipeline_fits_and_returns_finite_predictions():
    features = make_feature_table(load_monthly_series(DATA_FILE))
    training_data, test_data = split_by_time(features)
    selected_alpha, _ = select_ridge_alpha(training_data)
    model = build_forecasting_model(selected_alpha)
    model.fit(training_data.loc[:, MODEL_FEATURES], training_data["actual_sst_c"])
    predictions = model.predict(test_data.loc[:, MODEL_FEATURES])

    assert len(predictions) == len(test_data)
    assert np.isfinite(predictions).all()

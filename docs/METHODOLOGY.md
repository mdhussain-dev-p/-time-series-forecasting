# Methodology

## Research question

Does a Ridge regression model improve one-step-ahead forecasts of monthly Niño 1+2 sea-surface temperature compared with simple persistence, seasonal-naive and training-monthly-mean baselines?

## Target and predictors

The target is the observed temperature in month *t*. Predictors are available before that month:

- temperature at *t − 1*, *t − 2*, *t − 3*, *t − 6* and *t − 12*;
- sine and cosine representations of the target calendar month.

The first 12 observations are required to construct the longest lag. The target value is not included among the predictors.

## Chronological split

The 720 rows remaining after feature creation are kept in date order. The first 576 rows (January 1951–December 1998) form the training period. The final 144 rows (January 1999–December 2010) form the final test period. No random shuffle is used.

## Baselines

- **Persistence:** the previous month's observed temperature.
- **Seasonal naive:** the observed temperature from the same month one year earlier.
- **Training monthly mean:** the mean temperature for each calendar month, calculated using only target observations in the training period and then applied to the test dates.

The baselines are fixed comparisons; they do not use the final test outcomes to estimate their values.

## Ridge model and alpha selection

The Ridge pipeline standardises predictors and then fits `Ridge`. The regularisation value is not assigned by assumption. It is selected from `0.01`, `0.1`, `1`, `10`, `100` and `1000` using five expanding time-series validation splits within the training period. Each validation window contains 60 months. The alpha with the lowest mean validation MAE is selected, then the pipeline is refitted on all 576 training rows. The final 144-month test period is not involved in alpha selection.

## Forecast timing

The reported evaluation uses **one-step-ahead forecasts using observed history; no recursive prediction**. At each test date, lag features can include actual observations from earlier dates, including earlier dates in the test period. The fitted Ridge model does not feed its own previous predictions back as later lag values. A recursive multi-month forecast would use model-generated earlier values and may be less accurate as errors accumulate; that forecast setup is not evaluated here.

## Metrics

MAE is the average absolute error. RMSE is the square root of the mean squared error. Both are reported in degrees Celsius and lower values are better.

## Uncertainty for MAE differences

For each baseline, the monthly absolute error from Ridge is paired with the monthly absolute error from that baseline. The reported difference is Ridge absolute error minus baseline absolute error, averaged across the test dates. A negative value favours Ridge.

A circular moving-block bootstrap resamples paired monthly error differences in contiguous 12-month blocks, using 5,000 replicates and a fixed random seed (2026). The 2.5th and 97.5th percentiles form the approximate 95% interval. Resampling paired blocks preserves local month-to-month ordering better than independently resampling each month. This interval describes uncertainty in the paired test-period MAE difference; it is not a substitute for testing on multiple independent holdout periods.


## How to read the uncertainty interval

The bootstrap resamples paired monthly error differences in circular blocks of 12 consecutive months. Pairing keeps the two models' errors for a given month together, while the blocks retain some local month-to-month structure. The interval is an approximate description of uncertainty within this particular test window. It is not a prediction interval for future temperatures and it does not show how results would vary across entirely different test periods. The three comparisons are reported separately and are not adjusted for multiple comparisons.

# Methodology

## Forecasting question

The experiment asks whether a Ridge regression model can improve on a
persistence-style baseline for one-month-ahead sea-surface temperature
forecasts in the bundled Niño 1+2 series.

## Target and features

The target is the temperature in month *t*. The model is only given information
that would be available before that month:

- temperature at *t − 1*
- temperature at *t − 2*
- temperature at *t − 3*
- temperature at *t − 6*
- temperature at *t − 12*
- sine and cosine representations of the target month's calendar position

The sine/cosine terms represent the repeating nature of the calendar without
treating December and January as far apart. The target temperature itself is
kept separate from the feature columns.

## Train/test split

The lag of 12 months means the first 12 observations do not have a complete
feature row. The remaining 720 rows are split in chronological order: 576 for
training and 144 for testing. No random shuffle is used.

## Forecast timing

The model is fitted once on the training rows. For each month in the test
period, earlier observed temperatures are available as lagged inputs. This is
a rolling one-month-ahead setup. It is not equivalent to predicting the whole
test period from information available only at the end of 1998.

## Model and baseline

The baseline uses the previous month's observed temperature as the next
month's forecast. The model is a pipeline containing `StandardScaler` and
`Ridge(alpha=10.0)`. The pipeline ensures the scaling parameters are learnt
from training data only.

## Evaluation

Both forecasts are evaluated on the same 144 test months using MAE and RMSE.
The test period is not used to choose the Ridge regularisation value.

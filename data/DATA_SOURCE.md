# Dataset source and attribution

The file `elnino_monthly.csv` is a date-indexed version of the monthly El Niño
sea-surface temperature observations distributed with the `statsmodels`
El Niño dataset. The original observations are attributed to the National
Oceanic and Atmospheric Administration (NOAA), National Weather Service,
ERSST.V3B dataset for the Niño 1+2 region.

The region covers 0–10° South and 90–80° West. The supplied series contains
monthly average sea-surface temperature values in degrees Celsius from
January 1950 through December 2010 (732 observations).

The statsmodels dataset documentation identifies the data as public domain.
The CSV in this project has been reshaped into two columns (`date` and
`sst_c`) to make it easier to work with in pandas; the observation values have
not been intentionally altered.

## Sources

- statsmodels dataset description: https://www.statsmodels.org/stable/datasets/generated/elnino.html
- statsmodels data source code: https://github.com/statsmodels/statsmodels/blob/main/statsmodels/datasets/elnino/data.py
- NOAA Climate Prediction Center indices: https://www.cpc.ncep.noaa.gov/data/indices/

Source details checked on 7 November 2026. Consult the source pages for the
original metadata and any updates to the source data.

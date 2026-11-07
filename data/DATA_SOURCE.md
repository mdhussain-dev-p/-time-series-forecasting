# Dataset source and verification

The file `elnino_monthly.csv` contains monthly average sea-surface temperatures in degrees Celsius for the Niño 1+2 region (0–10° South, 90–80° West), from January 1950 through December 2010. It has 732 monthly records and two columns: `date` and `sst_c`.

The observations are reshaped from the El Niño dataset distributed by statsmodels. The upstream statsmodels dataset metadata attributes the observations to the National Oceanic and Atmospheric Administration's National Weather Service ERSST.V3B dataset for Niño 1+2. That metadata explicitly states: “This data is in the public domain.” This attribution and public-domain statement are reproduced from the upstream metadata; this project does not make a separate legal determination about other materials linked from NOAA.

## Value verification

On 8 November 2026, the bundled 732-row CSV was compared with all monthly values returned by `statsmodels.datasets.elnino` version 0.14.6. The years and months were expanded from the upstream 61-by-12 table into monthly dates. All 732 temperature values matched exactly at the precision supplied by statsmodels.

Spot checks include:

| Date | Bundled value (°C) | Upstream value (°C) |
|---|---:|---:|
| 1950-01-01 | 23.11 | 23.11 |
| 1950-02-01 | 24.20 | 24.20 |
| 2010-12-01 | 22.07 | 22.07 |

The CSV is a reshaped copy of these observations, not a newly measured dataset. No interpolation or synthetic values were added.

## Sources

- [statsmodels El Niño dataset documentation](https://www.statsmodels.org/stable/datasets/generated/elnino.html)
- [statsmodels upstream El Niño metadata and source attribution](https://github.com/statsmodels/statsmodels/blob/main/statsmodels/datasets/elnino/data.py)
- [NOAA Climate Prediction Center indices](https://www.cpc.ncep.noaa.gov/data/indices/)

Verification date: 8 November 2026.

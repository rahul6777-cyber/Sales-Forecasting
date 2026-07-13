# Sales Forecasting — Report

## Data
1,095 days (3 years, 2023-01-01 to 2025-12-30) of synthetic daily retail
sales with a realistic composition: gradual upward trend, weekly
(weekday/weekend) seasonality, yearly seasonality peaking around the
holidays, random promotion days (~8% of days, +$180 avg lift), and an
explicit Nov 20–Dec 24 holiday-shopping boost, plus noise.

*(No internet access in this sandbox to pull a real retail dataset, and the
`statsmodels`/`prophet` packages aren't installable offline — see
"Methodology" below for how this was handled.)*

## EDA (`eda_overview.png`)
- Clear **weekly cycle**: certain days consistently outsell others.
- Clear **yearly cycle**: sales rise through Q4 and peak around the holidays.
- **Promotion days sell substantially more** than regular days, visible as a
  distinct distribution shift in the boxplot.

## Methodology
Rather than skip ARIMA/Prophet because the packages can't be installed
offline, this script implements the same underlying techniques from
scratch with numpy/pandas/sklearn:
- **"Prophet-style" model** — piecewise-linear trend + Fourier-series
  seasonality (weekly + yearly) fit via linear regression. This is, in
  simplified form, exactly the decomposition Facebook Prophet performs.
- **"Holt-Winters-style" model** — classic triple exponential smoothing
  (level + trend + weekly seasonal component), implemented directly from
  its update equations.

If `statsmodels`/`prophet` are available in your own environment, swap in
`ExponentialSmoothing(...)` or `Prophet().fit(...)` directly — the data
prep, 90-day holdout split, and evaluation/plotting code all stay the same.

## Forecast evaluation (last 90 days held out)

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| **Prophet-style (trend + Fourier)** | **42.7** | **56.2** | **5.1%** |
| Holt-Winters-style (triple exp. smoothing) | 82.5 | 96.5 | 9.4% |

The Prophet-style regression model outperformed Holt-Winters here, mainly
because it captures the yearly holiday seasonality explicitly via Fourier
terms, while the Holt-Winters implementation only models weekly
seasonality directly (annual seasonality would need a 365-day season
length, which needs far more history to estimate reliably).

## Key artifacts
- `eda_overview.png` — weekly/monthly/promotion patterns
- `forecast_vs_actual.png` — full history + 90-day forecast vs. actual
- `forecast_detail.png` — zoomed-in view of the forecast window
- `forecast_model_comparison.csv` — accuracy metrics
- `daily_sales_data.csv` — the dataset used

## Recommendations
1. Use the Prophet-style model as the primary forecaster; it captures both
   weekly and holiday-season effects well (5.1% MAPE).
2. Feed a promotion calendar into the model as an explicit regressor (this
   version treats promotions as noise) — this would likely tighten
   forecasts further around promo periods.
3. Re-fit monthly as new data arrives, and consider a real ARIMA/Prophet
   library in production for automatic changepoint detection and confidence
   intervals.

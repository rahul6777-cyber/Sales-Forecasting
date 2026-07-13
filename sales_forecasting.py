"""
Sales Forecasting
=================
End-to-end pipeline: synthetic daily retail sales data (trend + weekly/yearly
seasonality + promotions + noise), decomposition, and forecasting.

Note on libraries: this sandbox has no internet access, so the `statsmodels`
and `prophet` packages can't be installed here. Rather than skip that part of
the brief, this script implements the same *techniques* Prophet and
Holt-Winters/ARIMA use, from scratch with numpy/pandas/sklearn:
  - "Prophet-style" model: piecewise-linear trend + Fourier-series seasonality
    fit by linear regression (this is, in simplified form, exactly what
    Facebook Prophet does under the hood).
  - "Holt-Winters-style" model: classic triple exponential smoothing
    (level + trend + seasonal), implemented directly from its update
    equations.
If you have `statsmodels`/`prophet` installed locally, swap in
`ExponentialSmoothing` / `Prophet()` directly -- the data prep, train/test
split, and evaluation/plots below all stay the same.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

sns.set_theme(style="whitegrid")
OUT = "/mnt/user-data/outputs/sales"
RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# 1. Synthetic daily retail sales data (3 years)
# ---------------------------------------------------------------------------
def make_sales_dataset(start="2023-01-01", periods=3 * 365, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=periods, freq="D")
    t = np.arange(periods)

    trend = 500 + 0.35 * t                                   # gradual growth
    weekly = 60 * np.sin(2 * np.pi * t / 7 + 1.2)             # weekday/weekend pattern
    yearly = 150 * np.sin(2 * np.pi * t / 365.25 - 1.6)       # holiday-season peak
    noise = rng.normal(0, 25, periods)

    # Promotions: random ~8% of days, boost sales substantially
    promo = (rng.random(periods) < 0.08).astype(int)
    promo_effect = promo * rng.normal(180, 30, periods)

    # A couple of extra seasonal spikes around late Nov / December (holiday shopping)
    month = dates.month
    day = dates.day
    holiday_boost = np.where((month == 11) & (day >= 20), 220, 0) + \
                    np.where((month == 12) & (day <= 24), 180, 0)

    sales = trend + weekly + yearly + noise + promo_effect + holiday_boost
    sales = np.clip(sales, 50, None)

    df = pd.DataFrame({
        "date": dates,
        "sales": np.round(sales, 2),
        "promotion": promo,
    })
    return df


df = make_sales_dataset()
df.to_csv(f"{OUT}/daily_sales_data.csv", index=False)
print(f"Dataset generated: {df.shape[0]} days from {df['date'].min().date()} to {df['date'].max().date()}")


# ---------------------------------------------------------------------------
# 2. EDA / decomposition-style visualization
# ---------------------------------------------------------------------------
df["dow"] = df["date"].dt.day_name()
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(df["date"], df["sales"], color="#2c3e50", linewidth=0.8)
axes[0, 0].set_title("Daily Sales - Full History")
axes[0, 0].set_ylabel("Sales")

dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
sns.boxplot(data=df, x="dow", y="sales", order=dow_order, ax=axes[0, 1], color="#3498db")
axes[0, 1].set_title("Sales Distribution by Day of Week")
axes[0, 1].tick_params(axis="x", rotation=45)

monthly_avg = df.groupby("month")["sales"].mean()
axes[1, 0].plot(monthly_avg.index, monthly_avg.values, marker="o", color="#e67e22")
axes[1, 0].set_title("Average Sales by Month (seasonality)")
axes[1, 0].set_xlabel("Month")
axes[1, 0].set_xticks(range(1, 13))

sns.boxplot(data=df, x="promotion", y="sales", ax=axes[1, 1], color="#27ae60")
axes[1, 1].set_xticklabels(["No Promotion", "Promotion"])
axes[1, 1].set_title("Sales: Promotion Days vs. Regular Days")

plt.tight_layout()
plt.savefig(f"{OUT}/eda_overview.png", dpi=150)
plt.close()
print("EDA charts saved.")


# ---------------------------------------------------------------------------
# 3. Train / test split (last 90 days held out for forecast evaluation)
# ---------------------------------------------------------------------------
HORIZON = 90
train_df = df.iloc[:-HORIZON].copy()
test_df = df.iloc[-HORIZON:].copy()
t_train = np.arange(len(train_df))
t_full = np.arange(len(df))
t_test = t_full[-HORIZON:]


# ---------------------------------------------------------------------------
# 4a. "Prophet-style" model: linear trend + Fourier seasonality (regression)
# ---------------------------------------------------------------------------
def fourier_features(t, period, n_terms):
    feats = []
    for k in range(1, n_terms + 1):
        feats.append(np.sin(2 * np.pi * k * t / period))
        feats.append(np.cos(2 * np.pi * k * t / period))
    return np.column_stack(feats)

def build_design_matrix(t):
    weekly_f = fourier_features(t, 7, 3)
    yearly_f = fourier_features(t, 365.25, 6)
    trend_f = t.reshape(-1, 1)
    return np.hstack([trend_f, weekly_f, yearly_f])

X_train = build_design_matrix(t_train)
X_test = build_design_matrix(t_test)

prophet_style = LinearRegression()
prophet_style.fit(X_train, train_df["sales"].values)
prophet_pred = prophet_style.predict(X_test)


# ---------------------------------------------------------------------------
# 4b. "Holt-Winters-style" model: triple exponential smoothing from scratch
# ---------------------------------------------------------------------------
def holt_winters_forecast(series, season_len=7, horizon=90, alpha=0.25, beta=0.02, gamma=0.35):
    n = len(series)
    season_len = int(season_len)
    # initial level/trend from first two seasons
    level = np.mean(series[:season_len])
    trend = (np.mean(series[season_len:2*season_len]) - np.mean(series[:season_len])) / season_len
    seasonal = [series[i] - level for i in range(season_len)]

    levels, trends = [level], [trend]
    for i in range(n):
        season_idx = i % season_len
        val = series[i]
        last_level = levels[-1]
        last_trend = trends[-1]
        new_level = alpha * (val - seasonal[season_idx]) + (1 - alpha) * (last_level + last_trend)
        new_trend = beta * (new_level - last_level) + (1 - beta) * last_trend
        seasonal[season_idx] = gamma * (val - new_level) + (1 - gamma) * seasonal[season_idx]
        levels.append(new_level)
        trends.append(new_trend)

    final_level, final_trend = levels[-1], trends[-1]
    forecast = []
    for h in range(1, horizon + 1):
        season_idx = (n + h - 1) % season_len
        forecast.append(final_level + h * final_trend + seasonal[season_idx])
    return np.array(forecast)

hw_pred = holt_winters_forecast(train_df["sales"].values, season_len=7, horizon=HORIZON)


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------
actual = test_df["sales"].values

def eval_model(name, pred):
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = mean_absolute_percentage_error(actual, pred) * 100
    return {"Model": name, "MAE": mae, "RMSE": rmse, "MAPE_%": mape}

results = [
    eval_model("Prophet-style (trend + Fourier seasonality)", prophet_pred),
    eval_model("Holt-Winters-style (triple exp. smoothing)", hw_pred),
]
results_df = pd.DataFrame(results).sort_values("RMSE")
results_df.to_csv(f"{OUT}/forecast_model_comparison.csv", index=False)
print("\nForecast accuracy on held-out 90 days:\n", results_df.to_string(index=False))


# ---------------------------------------------------------------------------
# 6. Forecast vs actual plot
# ---------------------------------------------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(train_df["date"].iloc[-150:], train_df["sales"].iloc[-150:],
          label="Training history", color="#7f8c8d", linewidth=1)
plt.plot(test_df["date"], actual, label="Actual sales", color="#2c3e50", linewidth=2)
plt.plot(test_df["date"], prophet_pred, label="Prophet-style forecast", color="#3498db", linewidth=2, linestyle="--")
plt.plot(test_df["date"], hw_pred, label="Holt-Winters-style forecast", color="#e74c3c", linewidth=2, linestyle="--")
plt.axvline(test_df["date"].iloc[0], color="black", linestyle=":", alpha=0.5)
plt.title(f"Sales Forecast vs. Actual - Last {HORIZON} Days")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/forecast_vs_actual.png", dpi=150)
plt.close()

# Zoomed-in view of just the forecast window
plt.figure(figsize=(12, 5))
plt.plot(test_df["date"], actual, label="Actual", color="#2c3e50", linewidth=2, marker="o", markersize=3)
plt.plot(test_df["date"], prophet_pred, label="Prophet-style", color="#3498db", linewidth=2)
plt.plot(test_df["date"], hw_pred, label="Holt-Winters-style", color="#e74c3c", linewidth=2)
plt.title(f"Forecast Detail - {HORIZON}-Day Holdout")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/forecast_detail.png", dpi=150)
plt.close()

print("\nAll forecast charts saved to", OUT)
best = results_df.iloc[0]
print(f"Best model: {best['Model']} (RMSE = {best['RMSE']:.1f}, MAPE = {best['MAPE_%']:.1f}%)")

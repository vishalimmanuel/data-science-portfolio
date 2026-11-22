"""
Project 6: Chennai Water Crisis - Reservoir Analysis and Forecast
-----------------------------------------------------------------
Chennai gets its drinking water mainly from four reservoirs: Poondi, Cholavaram, Redhills
and Chembarambakkam. In June 2019 the city nearly ran out of water ("Day Zero").

This project asks:
  1. How did total reservoir storage change from 2004 to 2019?
  2. How strongly does the monsoon (Oct-Dec rain) drive storage?
  3. Can we forecast storage 30 days ahead, and is that better than a simple guess?

Data: daily reservoir storage (million cubic feet) and daily rainfall (mm) for the four
reservoirs, from the Chennai Metropolitan Water Supply and Sewerage Board.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

BASE = "https://raw.githubusercontent.com/krish3010/ChennaiWaterManagement/master/"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42
RESERVOIRS = ["POONDI", "CHOLAVARAM", "REDHILLS", "CHEMBARAMBAKKAM"]
sns.set_theme(style="whitegrid")

# 1. LOAD AND CLEAN -----------------------------------------------------
levels = pd.read_csv(BASE + "chennai_reservoir_levels.csv")
rain = pd.read_csv(BASE + "chennai_reservoir_rainfall.csv")
for d in (levels, rain):
    d["Date"] = pd.to_datetime(d["Date"], format="%d-%m-%Y")
    d.set_index("Date", inplace=True)
    d.sort_index(inplace=True)

print(f"Levels: {levels.index.min().date()} to {levels.index.max().date()} ({len(levels)} days)")
full_range = pd.date_range(levels.index.min(), levels.index.max())
print(f"Missing days in the date range: {len(full_range) - len(levels)}")

df = pd.DataFrame({
    "storage": levels[RESERVOIRS].sum(axis=1),   # total storage, million cubic feet
    "rain": rain[RESERVOIRS].mean(axis=1),       # average rainfall across the 4 reservoirs, mm
}).reindex(full_range).interpolate(limit=5)      # fill short gaps only
df.index.name = "date"
print(f"Storage: min {df['storage'].min():.0f}, max {df['storage'].max():.0f} mcft")

# 2. ANALYSE ------------------------------------------------------------
# 2a. Storage over time, with each reservoir
fig, ax = plt.subplots(figsize=(11, 5))
levels[RESERVOIRS].plot.area(ax=ax, stacked=True, alpha=0.85, linewidth=0)
ax.set_title("Chennai reservoir storage, 2004-2019")
ax.set_ylabel("Storage (million cubic feet)")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(OUT / "storage_over_time.png", dpi=150)
plt.close()

# 2b. Lowest storage each year (shows the 2019 crisis)
yearly_min = df["storage"].groupby(df.index.year).min()
plt.figure(figsize=(9, 4.5))
colors = ["#e76f51" if y == yearly_min.idxmin() else "#2a9d8f" for y in yearly_min.index]
plt.bar(yearly_min.index, yearly_min.values, color=colors)
plt.title("Lowest total storage in each year")
plt.ylabel("Storage (mcft)")
plt.tight_layout()
plt.savefig(OUT / "yearly_minimum.png", dpi=150)
plt.close()
print("\nLowest storage by year:")
print(yearly_min.round(0).astype(int).to_string())

# 2c. Seasonality: which months are rainy and which have the most water?
monthly = df.groupby(df.index.month).agg(rain=("rain", "sum"), storage=("storage", "mean"))
monthly["rain"] = monthly["rain"] / df.index.year.nunique()  # average rain per year in that month
fig, ax1 = plt.subplots(figsize=(9, 4.5))
ax1.bar(monthly.index, monthly["rain"], color="#8ecae6", label="avg rainfall (mm)")
ax1.set_ylabel("Average monthly rainfall (mm)")
ax2 = ax1.twinx()
ax2.plot(monthly.index, monthly["storage"], color="#e76f51", marker="o", label="avg storage")
ax2.set_ylabel("Average storage (mcft)")
ax2.grid(False)
ax1.set_xticks(range(1, 13))
ax1.set_xlabel("Month")
plt.title("Rain peaks in Oct-Dec (northeast monsoon); storage follows a little later")
plt.tight_layout()
plt.savefig(OUT / "seasonality.png", dpi=150)
plt.close()

# 2d. Does more monsoon rain mean more water the next summer?
year_stats = []
for year in range(2004, 2019):
    monsoon_rain = df.loc[f"{year}-10-01":f"{year}-12-31", "rain"].sum()
    next_year_min = df.loc[f"{year + 1}-01-01":f"{year + 1}-12-31", "storage"].min()
    year_stats.append((year, monsoon_rain, next_year_min))
ys = pd.DataFrame(year_stats, columns=["year", "monsoon_rain_mm", "next_year_min_storage"])
corr = ys["monsoon_rain_mm"].corr(ys["next_year_min_storage"])
print(f"\nCorrelation between Oct-Dec rain and the NEXT year's lowest storage: {corr:.2f}")
plt.figure(figsize=(6.5, 5))
plt.scatter(ys["monsoon_rain_mm"], ys["next_year_min_storage"])
for _, r in ys.iterrows():
    plt.annotate(int(r["year"]), (r["monsoon_rain_mm"], r["next_year_min_storage"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
plt.xlabel("Oct-Dec rainfall (mm, summed over days)")
plt.ylabel("Next year's lowest storage (mcft)")
plt.title(f"Monsoon rain vs next year's low point (correlation {corr:.2f})")
plt.tight_layout()
plt.savefig(OUT / "monsoon_vs_next_year_low.png", dpi=150)
plt.close()

# 3. FORECAST STORAGE 30 DAYS AHEAD ------------------------------------
HORIZON = 30
feat = pd.DataFrame(index=df.index)
feat["storage_now"] = df["storage"]
feat["storage_7d_ago"] = df["storage"].shift(7)
feat["storage_30d_ago"] = df["storage"].shift(30)
feat["rain_7d"] = df["rain"].rolling(7).sum()
feat["rain_30d"] = df["rain"].rolling(30).sum()
feat["rain_90d"] = df["rain"].rolling(90).sum()
feat["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
feat["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
feat["target"] = df["storage"].shift(-HORIZON)
feat = feat.dropna()

# Time-based split: learn from the past, test on the future (never shuffle time series!).
train = feat[feat.index < "2016-01-01"]
test = feat[feat.index >= "2016-01-01"]
cols = [c for c in feat.columns if c != "target"]
print(f"\nForecast: train {train.index.min().date()}..{train.index.max().date()} ({len(train)} days), "
      f"test {test.index.min().date()}..{test.index.max().date()} ({len(test)} days)")

baseline = test["storage_now"]  # simplest guess: "30 days from now = today"
results = {"Baseline (same as today)": mean_absolute_error(test["target"], baseline)}

ridge = Ridge(alpha=1.0).fit(train[cols], train["target"])
results["Ridge regression"] = mean_absolute_error(test["target"], ridge.predict(test[cols]))

rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE)
# Predict the CHANGE in storage (not the level), so the model can still work when storage
# goes beyond what it saw in training.
rf.fit(train[cols], train["target"] - train["storage_now"])
rf_pred = test["storage_now"] + rf.predict(test[cols])
results["Random Forest (predicts change)"] = mean_absolute_error(test["target"], rf_pred)

print("\nMean absolute error of the 30-day-ahead forecast (mcft, lower is better):")
for k, v in results.items():
    print(f"  {k:<34} {v:,.0f}")

best_name = min(results, key=results.get)
(OUT / "results.txt").write_text(
    f"Lowest storage by year (mcft):\n{yearly_min.round(0).astype(int).to_string()}\n\n"
    f"Corr(Oct-Dec rain, next year's lowest storage) = {corr:.2f}\n\n"
    f"30-day-ahead forecast MAE (mcft), test period {test.index.min().date()} to {test.index.max().date()}:\n"
    + "\n".join(f"  {k}: {v:,.0f}" for k, v in results.items())
    + f"\n\nBest: {best_name}\n"
)

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(test.index, test["target"], label="actual (30 days later)", color="black", linewidth=1.5)
ax.plot(test.index, rf_pred, label="Random Forest forecast", color="#e76f51", linewidth=1)
ax.plot(test.index, baseline, label="baseline (= today)", color="#8ecae6", linewidth=1, alpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_ylabel("Storage (mcft)")
ax.set_title("30-day-ahead storage forecast on unseen data (2016-2019)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "forecast_vs_actual.png", dpi=150)
plt.close()
print(f"\nCharts and results saved in: {OUT}")

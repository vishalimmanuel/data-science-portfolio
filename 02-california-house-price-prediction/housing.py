"""
Project 2: California House Price Prediction (regression)
---------------------------------------------------------
Goal: predict the median house value of a California district from census data.

Steps:
  1. Load data
  2. Explore data (charts + correlations)
  3. Feature engineering (create more useful columns)
  4. Train and compare models, tune the best one
  5. Evaluate and explain (feature importance, prediction error)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42
sns.set_theme(style="whitegrid")

# 1. LOAD ---------------------------------------------------------------
data = fetch_california_housing(as_frame=True)
df = data.frame.rename(columns={"MedHouseVal": "price"})  # price is in $100,000s
print(f"Loaded {df.shape[0]} districts, {df.shape[1]} columns")
print(df.describe().T[["mean", "min", "max"]].round(2))
print("\nMissing values:", int(df.isna().sum().sum()))

# 2. EXPLORE ------------------------------------------------------------
plt.figure(figsize=(7, 4))
sns.histplot(df["price"], bins=50)
plt.title("Distribution of median house value ($100,000s)")
plt.tight_layout()
plt.savefig(OUT / "price_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation between columns")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(df["Longitude"], df["Latitude"], c=df["price"], s=4, cmap="viridis")
plt.colorbar(label="Median house value ($100,000s)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("House prices across California (coast = expensive)")
plt.tight_layout()
plt.savefig(OUT / "price_map.png", dpi=150)
plt.close()

# 3. FEATURE ENGINEERING -----------------------------------------------
# New columns that carry more meaning than the raw ones.
df["rooms_per_person"] = df["AveRooms"] / df["AveOccup"]
df["bedroom_ratio"] = df["AveBedrms"] / df["AveRooms"]

# The price column is capped at 5.0 ($500,000) in this dataset. Those rows hide the true price,
# so we remove them for a cleaner model. (A real-world decision worth mentioning in interviews.)
capped = (df["price"] >= 5.0).sum()
df = df[df["price"] < 5.0].reset_index(drop=True)
print(f"\nRemoved {capped} capped rows -> {len(df)} rows left")

X = df.drop(columns="price")
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# 4. TRAIN AND COMPARE --------------------------------------------------
models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Random Forest": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}

print("\nCross-validation R^2 (5 folds, higher is better):")
cv_scores = {}
for name, model in models.items():
    s = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    cv_scores[name] = s.mean()
    print(f"  {name:<20} {s.mean():.3f}")

# Tune the winner a little with grid search.
best_name = max(cv_scores, key=cv_scores.get)
print(f"\nTuning: {best_name}")
if best_name == "Gradient Boosting":
    grid = {"n_estimators": [200, 400], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
    search = GridSearchCV(models[best_name], grid, cv=3, scoring="r2", n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print("Best settings:", search.best_params_)
else:
    best_model = models[best_name].fit(X_train, y_train)

# 5. EVALUATE -----------------------------------------------------------
pred = best_model.predict(X_test)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred) * 100_000
rmse = root_mean_squared_error(y_test, pred) * 100_000
print(f"\nTest set: R^2 = {r2:.3f} | typical error (MAE) = ${mae:,.0f} | RMSE = ${rmse:,.0f}")

(OUT / "results.txt").write_text(
    "Cross-validation R^2:\n"
    + "\n".join(f"  {n}: {s:.3f}" for n, s in cv_scores.items())
    + f"\n\nBest model: {best_name}\n"
    + f"Test R^2: {r2:.3f}\nTest MAE: ${mae:,.0f}\nTest RMSE: ${rmse:,.0f}\n"
)

plt.figure(figsize=(6, 6))
plt.scatter(y_test * 100_000, pred * 100_000, s=5, alpha=0.4)
lim = [0, 500_000]
plt.plot(lim, lim, "r--", label="perfect prediction")
plt.xlabel("Actual price ($)")
plt.ylabel("Predicted price ($)")
plt.title(f"{best_name}: actual vs predicted")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "actual_vs_predicted.png", dpi=150)
plt.close()

# Which columns matter most?
if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values()
    plt.figure(figsize=(7, 5))
    imp.plot.barh()
    plt.title("Feature importance")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close()
    print("\nTop 3 features:", ", ".join(imp.sort_values(ascending=False).index[:3]))

print(f"\nCharts and results saved in: {OUT}")

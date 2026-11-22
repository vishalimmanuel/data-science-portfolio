"""
Project 5: Chennai House Price Prediction (messy real-world data)
-----------------------------------------------------------------
Goal: predict the sale price of a property in Chennai from its location, size and quality.

What makes this project interesting: the raw data is MESSY. The same area is spelled many
ways ("Chrompet", "Chrompt", "Chrmpet"), so most of the work is cleaning, then modelling.

Steps:
  1. Load data
  2. Clean the messy text columns (typo fixing)
  3. Feature engineering (building age, sale year)
  4. Avoid data leakage (drop columns calculated FROM the price)
  5. Explore (price by area)
  6. Train and compare models, evaluate on hidden data
"""

from difflib import get_close_matches
from pathlib import Path
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = (
    "https://raw.githubusercontent.com/aprilslippedaway/Chennai-House-Price-Prediction-Dataset/main/"
    + quote("Chennai houseing sale.csv")
)
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42
LAKH = 100_000  # Indian unit: 1 lakh = 100,000 rupees
sns.set_theme(style="whitegrid")

# 1. LOAD ---------------------------------------------------------------
df = pd.read_csv(DATA_URL)
print(f"Loaded {df.shape[0]} properties, {df.shape[1]} columns")
print("Raw unique AREA spellings:", df["AREA"].nunique())

# 2. CLEAN --------------------------------------------------------------
# For each text column we list the correct spellings, then map every typo to the closest one.
CORRECT = {
    "AREA": ["Karapakkam", "Anna Nagar", "Adyar", "Velachery", "Chrompet", "KK Nagar", "T Nagar"],
    "SALE_COND": ["AbNormal", "Family", "Partial", "AdjLand", "Normal Sale"],
    "PARK_FACIL": ["Yes", "No"],
    "BUILDTYPE": ["Commercial", "Others", "House"],
    "UTILITY_AVAIL": ["AllPub", "ELO", "NoSewa"],
    "STREET": ["Paved", "Gravel", "No Access"],
}


def fix_spelling(value, correct_list):
    """Lowercase, remove spaces, then pick the closest correct spelling."""
    key = str(value).lower().replace(" ", "")
    keys = {c.lower().replace(" ", ""): c for c in correct_list}
    match = get_close_matches(key, keys.keys(), n=1, cutoff=0.6)
    return keys[match[0]] if match else None


for col, correct_list in CORRECT.items():
    before = df[col].nunique()
    df[col] = df[col].apply(lambda v: fix_spelling(v, correct_list))
    print(f"  {col:<14} {before} spellings -> {df[col].nunique()} clean values")
    assert df[col].isna().sum() == 0, f"Unmatched values in {col}"

# 3. FEATURE ENGINEERING ------------------------------------------------
df["DATE_SALE"] = pd.to_datetime(df["DATE_SALE"], dayfirst=True)
df["DATE_BUILD"] = pd.to_datetime(df["DATE_BUILD"], dayfirst=True)
df["sale_year"] = df["DATE_SALE"].dt.year
df["building_age"] = (df["DATE_SALE"] - df["DATE_BUILD"]).dt.days / 365.25
print(f"\nBuilding age ranges from {df['building_age'].min():.0f} to {df['building_age'].max():.0f} years")

# 4. DATA LEAKAGE ------------------------------------------------------
# REG_FEE (registration fee) and COMMIS (commission) are calculated FROM the sale price.
# Using them would let the model "cheat", so we drop them.
print("Correlation of REG_FEE with price:", round(df["REG_FEE"].corr(df["SALES_PRICE"]), 2),
      "-> dropped to avoid leakage")
df["price_lakh"] = df["SALES_PRICE"] / LAKH

# 5. EXPLORE ------------------------------------------------------------
order = df.groupby("AREA")["price_lakh"].median().sort_values().index
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="AREA", y="price_lakh", order=order)
plt.ylabel("Sale price (lakh Rs)")
plt.title("Property prices by Chennai area")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(OUT / "price_by_area.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5.5))
sns.scatterplot(data=df, x="INT_SQFT", y="price_lakh", hue="AREA", s=12, alpha=0.6)
plt.xlabel("Interior area (sq ft)")
plt.ylabel("Sale price (lakh Rs)")
plt.title("Price vs size, coloured by area")
plt.tight_layout()
plt.savefig(OUT / "price_vs_size.png", dpi=150)
plt.close()

# 6. MODEL --------------------------------------------------------------
categorical = ["AREA", "SALE_COND", "PARK_FACIL", "BUILDTYPE", "UTILITY_AVAIL", "STREET", "MZZONE"]
numeric = ["INT_SQFT", "DIST_MAINROAD", "N_BEDROOM", "N_BATHROOM", "N_ROOM",
           "QS_ROOMS", "QS_BATHROOM", "QS_BEDROOM", "QS_OVERALL", "sale_year", "building_age"]

X = df[categorical + numeric]
y = df["price_lakh"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def make_pipeline(model, scale):
    num_steps = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", Pipeline(num_steps), numeric),
    ])
    return Pipeline([("prep", prep), ("model", model)])


models = {
    "Linear Regression": make_pipeline(LinearRegression(), scale=True),
    "Random Forest": make_pipeline(RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE), scale=False),
    "Gradient Boosting": make_pipeline(GradientBoostingRegressor(random_state=RANDOM_STATE), scale=False),
}

print("\nCross-validation R^2 (5 folds):")
cv = {}
for name, pipe in models.items():
    cv[name] = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()
    print(f"  {name:<20} {cv[name]:.3f}")

best_name = max(cv, key=cv.get)
best = models[best_name].fit(X_train, y_train)
pred = best.predict(X_test)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
median_price = y_test.median()
print(f"\nBest model: {best_name}")
print(f"Test R^2 = {r2:.3f} | typical error (MAE) = {mae:.1f} lakh Rs "
      f"({mae / median_price:.0%} of the median price of {median_price:.0f} lakh)")

(OUT / "results.txt").write_text(
    "Cross-validation R^2:\n" + "\n".join(f"  {n}: {s:.3f}" for n, s in cv.items())
    + f"\n\nBest model: {best_name}\nTest R^2: {r2:.3f}\nTest MAE: {mae:.1f} lakh Rs\n"
    f"Median price in test set: {median_price:.0f} lakh Rs\n"
)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred, s=6, alpha=0.4)
lim = [0, max(y_test.max(), pred.max())]
plt.plot(lim, lim, "r--", label="perfect prediction")
plt.xlabel("Actual price (lakh Rs)")
plt.ylabel("Predicted price (lakh Rs)")
plt.title(f"{best_name}: actual vs predicted")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "actual_vs_predicted.png", dpi=150)
plt.close()

# What matters most? (permutation importance works for any model)
from sklearn.inspection import permutation_importance

imp = permutation_importance(best, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
imp_s = pd.Series(imp.importances_mean, index=X.columns).sort_values()
plt.figure(figsize=(7, 6))
imp_s.plot.barh()
plt.title("Which features matter most (permutation importance)")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=150)
plt.close()
print("Top 3 features:", ", ".join(imp_s.sort_values(ascending=False).index[:3]))
print(f"\nCharts and results saved in: {OUT}")

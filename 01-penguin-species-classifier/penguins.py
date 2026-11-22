"""
Project 1: Palmer Penguins - Species Classification
---------------------------------------------------
Goal: predict a penguin's species (Adelie / Chinstrap / Gentoo) from body measurements.

Steps (each one is a section below):
  1. Load data
  2. Clean data
  3. Explore data (charts)
  4. Train and compare models
  5. Evaluate the best model
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw charts to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42  # makes results repeatable

# 1. LOAD ---------------------------------------------------------------
df = pd.read_csv(DATA_URL)
print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
print("\nMissing values per column:\n", df.isna().sum())

# 2. CLEAN --------------------------------------------------------------
# Only a few rows are missing measurements, so we drop them (and keep a note of how many).
before = len(df)
df = df.dropna().reset_index(drop=True)
print(f"\nDropped {before - len(df)} rows with missing values -> {len(df)} rows left")

# 3. EXPLORE ------------------------------------------------------------
sns.set_theme(style="whitegrid")

# 3a. How many penguins of each species?
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="species", hue="species", legend=False)
plt.title("Penguins per species")
plt.tight_layout()
plt.savefig(OUT / "species_counts.png", dpi=150)
plt.close()

# 3b. Which measurements separate the species?
pair = sns.pairplot(
    df,
    hue="species",
    vars=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
)
pair.figure.suptitle("Measurements by species", y=1.02)
pair.savefig(OUT / "pairplot.png", dpi=120)
plt.close("all")

# 4. TRAIN AND COMPARE MODELS ------------------------------------------
features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
X = df[features]
y = df["species"]

# Keep 20% of the data hidden until the very end, to test honestly.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

# 5-fold cross-validation on the training set: a fair way to compare models.
print("\nCross-validation accuracy (5 folds):")
scores = {}
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=5)
    scores[name] = cv.mean()
    print(f"  {name:<20} {cv.mean():.3f} (+/- {cv.std():.3f})")

best_name = max(scores, key=scores.get)
best_model = models[best_name].fit(X_train, y_train)
print(f"\nBest model: {best_name}")

# 5. EVALUATE ON HIDDEN TEST DATA --------------------------------------
pred = best_model.predict(X_test)
report = classification_report(y_test, pred)
print("\nTest set results:\n", report)

(OUT / "results.txt").write_text(
    "Cross-validation accuracy:\n"
    + "\n".join(f"  {n}: {s:.3f}" for n, s in scores.items())
    + f"\n\nBest model: {best_name}\n\nTest set results:\n{report}"
)

ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap="Blues")
plt.title(f"{best_name}: confusion matrix (test set)")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=150)
plt.close()

print(f"\nCharts and results saved in: {OUT}")

# Penguin Species Classifier

Predicts a penguin's species (Adelie, Chinstrap or Gentoo) from four body measurements using the Palmer Penguins dataset.

## What it does
1. Loads 344 penguin records and drops the 11 rows with missing values
2. Explores the data with charts (species counts, measurement pair plots)
3. Compares three models using 5-fold cross-validation: Logistic Regression, Decision Tree, Random Forest
4. Evaluates the best model on a hidden 20% test set

## Results
| Model | Cross-validation accuracy |
|---|---|
| Logistic Regression | 98.1% |
| Random Forest | 97.7% |
| Decision Tree | 95.8% |

The best model (Logistic Regression) scored 100% on the 67-row test set. This is a small, easy dataset, so a perfect test score is partly luck. The cross-validation figure (about 98%) is the more trustworthy number.

Charts and full results are in [`outputs/`](outputs).

## Run it
```bash
pip install -r requirements.txt
python penguins.py
```

## Skills shown
pandas, data cleaning, exploratory analysis, seaborn and matplotlib, scikit-learn pipelines, cross-validation, model comparison

## Data
[Palmer Penguins](https://github.com/mwaskom/seaborn-data) (public dataset, loaded from the seaborn-data repository).

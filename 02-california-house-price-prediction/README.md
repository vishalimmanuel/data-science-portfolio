# California House Price Prediction

Predicts the median house value of a California district from census data (income, house age, rooms, location and more).

## What it does
1. Loads 20,640 districts from scikit-learn's California Housing dataset
2. Explores prices with a distribution chart, correlation heatmap and a latitude/longitude price map
3. Creates two new features (`rooms_per_person`, `bedroom_ratio`)
4. Removes 992 rows where the price is capped at $500,000, since the true value is unknown
5. Compares Linear Regression, Random Forest and Gradient Boosting with 5-fold cross-validation
6. Evaluates the best model on a hidden 20% test set

## Results
| Model | Cross-validation R² |
|---|---|
| Random Forest | 0.772 |
| Gradient Boosting | 0.754 |
| Linear Regression | 0.613 |

On the test set, the Random Forest scored **R² = 0.774** with a typical error (MAE) of about **$31,000** (RMSE about $46,500). The most important features were median income, average occupancy and longitude.

Charts are in [`outputs/`](outputs), including the price map and actual-vs-predicted plot.

## Limitations
- The data is from the 1990 census, so the prices are old and not useful for real valuation
- Districts are averages, not individual houses

## Run it
```bash
pip install -r requirements.txt
python housing.py
```
The dataset downloads automatically the first time.

## Skills shown
pandas, exploratory analysis, feature engineering, handling capped data, regression, cross-validation, model comparison, feature importance

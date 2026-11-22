# Chennai House Price Prediction

Predicts the sale price of a property in Chennai (Adyar, Anna Nagar, Velachery, T Nagar, KK Nagar, Chrompet, Karapakkam) from its size, location, building type and quality scores.

The raw data is deliberately messy, so most of the work is **data cleaning**.

## What it does
1. Loads 7,109 property sales
2. **Cleans typos** in text columns by matching each value to the closest correct spelling. For example, 17 spellings of the area names (`Chrompt`, `Chrmpet`, `Ana Nagar`, `Karapakam`...) become the 7 real areas. Five other columns are cleaned the same way
3. Builds features: building age at sale and sale year
4. **Avoids data leakage:** `REG_FEE` and `COMMIS` (registration fee, commission) are calculated from the sale price (correlation 0.88 for the fee), so they are dropped. Using them would let the model cheat
5. Explores prices by area and size
6. Compares Linear Regression, Random Forest and Gradient Boosting with 5-fold cross-validation, then tests on hidden data

## Results
| Model | Cross-validation R² |
|---|---|
| Gradient Boosting | 0.988 |
| Random Forest | 0.980 |
| Linear Regression | 0.958 |

On the hidden test set, Gradient Boosting scored **R² = 0.989**, with a typical error of about **3 lakh Rs (3% of the median price of 102 lakh)**. The most important features were building type, interior area and location.

## An honest note about these scores
An R² this high is unusual for real property data. Once the typos are fixed, the dataset behaves very regularly, so it is probably generated or heavily structured. The scores show the pipeline works well. They should not be read as "this model can price real Chennai homes".

Charts are in [`outputs/`](outputs).

## Run it
```bash
pip install -r requirements.txt
python chennai_houses.py
```
The data downloads automatically from a public GitHub copy of the dataset ([source](https://github.com/aprilslippedaway/Chennai-House-Price-Prediction-Dataset)). The data is not stored in this repository.

## Skills shown
Real-world data cleaning, fuzzy text matching, date features, data-leakage awareness, scikit-learn pipelines (imputation, encoding, scaling), regression, cross-validation, permutation importance

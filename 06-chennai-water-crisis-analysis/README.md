# Chennai Water Crisis: Reservoir Analysis and Forecast

Chennai's drinking water comes mainly from four reservoirs: Poondi, Cholavaram, Redhills and Chembarambakkam. In June 2019 the city nearly ran out of water. This project analyses 15 years of daily reservoir data (Jan 2004 to Jun 2019) to see how that happened.

## Questions and answers

**1. How did storage change over time?**
Storage swings every year, from a few thousand million cubic feet (mcft) down to almost nothing. The lowest points were 71 mcft in 2017 and 26 mcft in 2019 (data ends 17 June 2019, so the 2019 low is only a partial year). The reservoirs have been close to empty several times in recent years: 2015 (272), 2016 (723), 2017 (71) and 2018 (869).

**2. Does the monsoon drive storage?**
The seasonal pattern is very clear. Rain peaks from October to December (northeast monsoon), storage is highest in December and lowest in August. But a good monsoon did **not** reliably mean more water the next year: the correlation between Oct-Dec rainfall and the following year's lowest storage was only **0.24**, based on 15 data points.

**3. Can we forecast storage 30 days ahead?**
I trained on 2004-2015 and tested on unseen data from 2016 to 2019, comparing models against a simple baseline of "30 days from now will equal today".

| Model | Error (MAE, mcft, lower is better) |
|---|---|
| Baseline (same as today) | **580** |
| Ridge regression | 618 |
| Random Forest (predicts change) | 661 |

**Neither model beat the baseline.** This is a genuine negative result. Rainfall alone, even with recent-history features, does not predict storage well here. The likely missing factors, such as water released, transferred or drawn from the reservoirs, are not in this dataset (I haven't tested that). A useful next step would be adding more data sources before building a more complex model.

Charts are in [`outputs/`](outputs).

## Run it
```bash
pip install -r requirements.txt
python water.py
```
The data downloads automatically from a public GitHub copy of the Chennai Metropolitan Water Supply and Sewerage Board data ([source](https://github.com/krish3010/ChennaiWaterManagement)). The data is not stored in this repository.

## Limitations
- Only 15 years of data, and the 2019 year is incomplete
- Rainfall is averaged across the four reservoirs, and the totals ignore each reservoir's capacity
- Correlations here do not prove cause and effect

## Skills shown
Time-series handling, date parsing, gap checking and interpolation, seasonal analysis, feature engineering with lags and rolling windows, time-based train/test splitting, baseline comparison, honest reporting of a negative result

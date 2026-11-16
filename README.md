# Retail Sales Forecasting: statistical, ML and deep-learning models

MSc Data Science portfolio project. **Question:** can LSTM beat ARIMA/SARIMA, Prophet and XGBoost on retail sales?
We do not assume it can. Research prototype, not a production system.

## Data and update frequency
* **Live source:** US Census retail sales via FRED API. **Monthly** (advance estimate about mid-month, later revised).
  Not real-time, not daily. Ingestion is incremental and idempotent; revisions are stored as new versions.
  Series IDs (`RSAFSNA`, `RSXFSN`) must be verified as NOT seasonally adjusted on the FRED site.
* **Static benchmark:** M5 Walmart (Kaggle), daily UNIT sales, not live. Pipeline written, not yet run on real data.

## Models (9 variants)
naive, seasonal_naive, arima, sarima, prophet, xgboost, xgboost_logdiff, lstm, lstm_logdiff.
`*_logdiff` learn log1p growth instead of the level and convert back to levels (handles trending series). Both versions are always reported.

## Measured results (FRED RSAFSNA, experiment c321273e, run on 2026-11-12, ONE seed, ONE split)
Test period June 2021 to September 2026. Log-diff variants and significance tests: **not measured yet** (run the commands below).

| Model | Test MAE | WAPE % |
|---|---|---|
| sarima | 19,455 | 2.89 |
| seasonal_naive | 32,650 | 4.85 |
| naive | 33,006 | 4.90 |
| arima | 37,523 | 5.57 |
| lstm | 69,994 | 10.39 |
| prophet | 87,383 | 12.97 |
| xgboost | 202,589 | 30.07 |

Preliminary. Do not generalise until seeds, significance tests and the log-diff comparison are done.

## Setup (Windows, uv)
```
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
copy .env.example .env        (then paste FRED_API_KEY)
```
Use `uv pip`, never bare `pip`.

## Run
```
python -m src.data.ingestion
python -m src.data.preprocessing
python -m src.run_experiments --dataset fred_monthly --series RSAFSNA          # add --quick for a smoke test
python -m src.evaluation.significance --dataset fred_monthly --series RSAFSNA --reference seasonal_naive
python -m src.run_seeds --dataset fred_monthly --series RSAFSNA --seeds 5
python -m src.evaluation.plots --dataset fred_monthly --series RSAFSNA
streamlit run dashboard/app.py
pytest
```
M5 (after placing `sales_train_evaluation.csv` and `calendar.csv` in `data/external/m5/`):
```
python -m src.data.m5_loader
python -m src.run_experiments --dataset m5_daily --quick      # SARIMA grid is slow on daily data
```

## Protocol (leakage control)
Chronological 70/15/15. Fit on train, tune on validation, score test. Rolling-origin blocks of `horizon` steps, history only before each block, no per-window refit.
Metrics: MAE, RMSE, sMAPE, WAPE, MAPE. Diebold-Mariano (HLN-corrected, Holm-adjusted) and multi-seed spread for stochastic models.
See `docs/LEAKAGE_AUDIT.md` and `docs/RESEARCH_PLAN.md`.

## Limitations
About 400 monthly points (small for LSTM); one split; test period reused; data revisions; structural break 2020-2026; limited LSTM tuning;
Prophet is not conditioned on recent actuals; saved models are fitted on train only; M5 target is units, not revenue.

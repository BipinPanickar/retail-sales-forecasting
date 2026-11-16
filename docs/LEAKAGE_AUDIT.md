# Leakage audit (living document)

Status legend: VERIFIED = automated test or direct measurement; REVIEWED = checked by code reading only;
OPEN = a known risk that cannot be removed, only disclosed.

| # | Check | Status | Evidence / where |
|---|---|---|---|
| 1 | Chronological split, no shuffling | VERIFIED | `tests/test_leakage.py::test_split_order_train_val_test`, `src/evaluation/backtest.py` |
| 2 | Each forecast window sees only history before its origin | VERIFIED | `tests/test_leakage.py` (spy model), `rolling_origin` |
| 3 | Rolling/lag features use data up to t-1 only | VERIFIED | `tests/test_features.py::test_no_future_leakage_in_features` |
| 4 | Recursive feature rows match training features | VERIFIED | `tests/test_features.py::test_row_builder_matches_vectorised_features` |
| 5 | Log-diff transform uses no learned parameters | VERIFIED | `tests/test_logdiff.py`, `src/models/logdiff.py` |
| 6 | LSTM scaler fitted on TRAIN only | REVIEWED | `LSTMForecaster.fit` (`MinMaxScaler().fit(train[["sales"]])`); no automated test (needs TensorFlow) |
| 7 | ARIMA/SARIMA d chosen by ADF on TRAIN only | REVIEWED | `choose_d(train["sales"])` |
| 8 | Hyperparameters chosen on VALIDATION only; test scored once per run | REVIEWED | every `select()` uses `rolling_origin` on validation; test scored in `run_experiments.py` after selection |
| 9 | Dashboard ACF/PACF/stationarity default to TRAIN only | REVIEWED | sidebar default in `dashboard/app.py` |
| 10 | Prophet fitted on TRAIN only | REVIEWED | `ProphetForecaster.fit_on(train)` |
| 11 | Exogenous inputs (M5 calendar/SNAP) known in advance | REVIEWED | `src/data/m5_loader.py` (not yet run on real M5 data) |

## Open risks (must be disclosed in the dissertation)

1. **Researcher degrees of freedom.** Prophet and the log-diff variants were added AFTER the first test results
   were seen. The validation results alone already showed the same pattern (XGBoost val MAE about 36k vs SARIMA about 19k),
   but the order of events is a legitimate criticism. Mitigation: report every variant that was run (plain AND log-diff),
   freeze the model list and settings now, and add no further variants before the final run.
2. **Test period reused.** Test metrics have been looked at several times. Mitigation: the live pipeline keeps ingesting
   new months; forecasts made now and scored on months that did not yet exist are a genuine out-of-sample check
   (store them with `python -m src.forecasting.forecast --save`).
3. **Data revisions.** The series used is the latest vintage; early points were revised after publication. Evaluating on
   revised values is mildly optimistic relative to real-time use. `load_fred_series(as_of=...)` gives a point-in-time view of
   what WE ingested, but true vintage data (ALFRED) is not used.
4. **Structural break.** The validation/test windows contain 2020-2026 (shock, inflation). Single split, single seed.
5. **Saved models are fitted on TRAIN only** and conditioned on later data without re-estimation (design decision).

## Not yet verified
- That `RSAFSNA` / `RSXFSN` are not seasonally adjusted (to be checked on the FRED series pages).
- M5 pipeline on real data.

# Research plan (draft). No citations are included: they must come from sources YOU have read and verified.

## Working question
Does an LSTM forecast retail sales better than ARIMA/SARIMA, Prophet and XGBoost, once the evaluation is leak-free and
the series level-shift problem is handled? (Not assumed.)

## Research questions and testable hypotheses
| RQ | Hypothesis (H1 = alternative) | Evidence we will use |
|---|---|---|
| RQ1 | Seasonal baselines/SARIMA beat non-seasonal ARIMA on monthly NSA retail sales | MAE/WAPE, Diebold-Mariano vs seasonal naive |
| RQ2 | XGBoost with lag/calendar features improves on baselines ONLY after the level shift is handled (log-diff) | plain vs `xgboost_logdiff`, DM test |
| RQ3 | LSTM does not significantly beat SARIMA on about 400 monthly points | multi-seed MAE, DM test |
| RQ4 | Best accuracy/cost trade-off is SARIMA or a simple model | MAE vs training time |
| RQ5 | Error grows with forecast step; the ranking can change by step | error-by-step plot |
| RQ6 | Ranking on M5 daily data differs from FRED monthly | M5 run (not yet done) |

## Objectives
1. Reproducible ingestion of regularly updated retail data (done, monthly, with revision history).
2. Leak-free comparison of 9 model variants under one protocol (done for FRED, preliminary, one seed).
3. Statistical validation: DM tests and seed variance (code ready; needs running).
4. Static daily benchmark (code ready; needs M5 download and running).
5. Honest write-up of limitations (see `docs/LEAKAGE_AUDIT.md`).

## Preliminary, measured (FRED RSAFSNA, experiment c321273e, one seed, one split)
SARIMA test MAE about 19.5k vs about 32.6k seasonal naive. XGBoost, LSTM, Prophet much worse in the PLAIN versions.
These are preliminary and are NOT yet a finding.

## Literature protocol (to do by you; I cannot browse, so I will not invent references)
- Databases: IEEE Xplore, ACM DL, ScienceDirect, SpringerLink, Google Scholar for cross-checking.
- Search strings to try: "retail sales forecasting" AND (SARIMA OR LSTM); "LSTM vs ARIMA" AND "monthly";
  "deep learning" "short time series" forecasting; "M5 competition" forecasting; "Diebold-Mariano" forecast comparison.
- Inclusion: peer-reviewed, empirical comparison on real retail/macro series, reports test protocol.
- Record for each paper: data size, frequency, split method, baselines used, metrics, whether leakage controls are described.
- A "research gap" is claimed ONLY if the table above shows no paper doing the same comparison under the same protocol.
  Do not claim novelty before that table exists.

## Threats to validity
Single dataset family; one split; test reused; data revisions; structural break; limited LSTM tuning (3 lookbacks, fixed architecture).

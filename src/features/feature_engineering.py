"""Lag / rolling / calendar features for XGBoost.

LEAKAGE RULE: the feature row for time t may use sales only up to t-1.
  lag_k(t)     = y[t-k]
  rmean_w(t)   = mean(y[t-w .. t-1])      (shift(1) before rolling)
Calendar variables and known-in-advance exogenous flags (events, SNAP) are allowed at time t.
make_features (vectorised, training) and feature_row (one step, recursive forecasting)
must give identical numbers - tests/test_features.py checks this.
"""
import numpy as np
import pandas as pd


def calendar_features(idx: pd.DatetimeIndex, freq: str) -> pd.DataFrame:
    c = pd.DataFrame(index=idx)
    c["month"] = idx.month
    c["quarter"] = idx.quarter
    if freq.upper().startswith("D"):
        c["dayofweek"] = idx.day_of_week
        c["dayofmonth"] = idx.day
        c["weekofyear"] = idx.isocalendar().week.astype(int).values
    return c


def make_features(df: pd.DataFrame, lags, windows, exog_cols, freq: str) -> pd.DataFrame:
    y = df["sales"]
    X = pd.DataFrame(index=df.index)
    for k in lags:
        X[f"lag_{k}"] = y.shift(k)
    past = y.shift(1)
    for w in windows:
        X[f"rmean_{w}"] = past.rolling(w).mean()
        X[f"rstd_{w}"] = past.rolling(w).std()
    X = X.join(calendar_features(df.index, freq))
    for c in exog_cols:
        X[c] = df[c]
    return X


def feature_row(y_hist, date, exog_row, lags, windows, exog_cols, freq: str, columns) -> pd.DataFrame:
    """One feature row for `date` given the list/array of sales BEFORE that date."""
    y = np.asarray(y_hist, float)
    row = {}
    for k in lags:
        row[f"lag_{k}"] = y[-k]
    for w in windows:
        row[f"rmean_{w}"] = y[-w:].mean()
        row[f"rstd_{w}"] = y[-w:].std(ddof=1)
    row.update(calendar_features(pd.DatetimeIndex([date]), freq).iloc[0].to_dict())
    for c in exog_cols:
        row[c] = exog_row[c]
    return pd.DataFrame([row])[columns]

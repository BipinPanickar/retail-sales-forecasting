"""Chronological splitting and leak-free rolling-origin evaluation.

Protocol (same for every model):
  * Models are fitted on TRAIN only. Hyperparameters are chosen on VALIDATION.
  * TEST is touched once, for the final comparison.
  * For each forecast window the model sees only actual data BEFORE the window origin
    (parameters are NOT refitted per window; only the history is extended).
  * Windows are non-overlapping blocks of `horizon` steps, so per-step errors
    show how accuracy changes with horizon (RQ5).
"""
import numpy as np
import pandas as pd

NON_EXOG = ("sales", "imputed")


def get_exog_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_EXOG]


def chronological_split(df: pd.DataFrame, train_frac: float, val_frac: float):
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return df.iloc[:n_train], df.iloc[n_train:n_train + n_val], df.iloc[n_train + n_val:]


def rolling_origin(model, df: pd.DataFrame, start: int, end: int, horizon: int) -> pd.DataFrame:
    """Forecast df.iloc[start:end] in blocks of `horizon`, using only data before each block."""
    exog = get_exog_cols(df)
    out = []
    for origin in range(start, end, horizon):
        h = min(horizon, end - origin)
        hist = df.iloc[:origin]
        fut = df.iloc[origin:origin + h]
        pred = np.asarray(model.predict(hist, fut[exog]), dtype=float)
        out.append(pd.DataFrame({"date": fut.index, "origin": hist.index[-1],
                                 "step": np.arange(1, h + 1),
                                 "actual": fut["sales"].values, "pred": pred}))
    return pd.concat(out, ignore_index=True)

"""Log-difference target wrapper (agreed decision: report BOTH plain and log-diff versions).

Why: tree models (XGBoost) and bounded-scaled neural nets (LSTM) cannot extrapolate beyond the
level range seen in training, which hurts on trending series. Here the inner model learns
r_t = log1p(y_t) - log1p(y_{t-1}) (growth), and forecasts are converted back to levels:
    y_{t+h} = expm1( log1p(y_t) + cumsum(r_{t+1..t+h}) )
log1p/expm1 (not log/exp) so zero-sales days (e.g. M5 Christmas Day) do not break the transform.

Leakage: the transform at time t uses only y_t and y_{t-1}; no parameters are learned from data.
The wrapper is evaluated on the ORIGINAL sales scale, so metrics are comparable with all other models.
"""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster, write_json


def to_logdiff(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 'sales' by its log1p-difference; the first row is dropped (no previous value)."""
    if (df["sales"] < 0).any():
        raise ValueError("log1p transform needs non-negative sales.")
    out = df.copy()
    out["sales"] = np.log1p(df["sales"]).diff()
    return out.iloc[1:]


def split_logdiff(train: pd.DataFrame, val: pd.DataFrame):
    """Transform train+val together (so the first validation row keeps its diff), then re-split."""
    full_t = to_logdiff(pd.concat([train, val]))
    return full_t.iloc[:len(train) - 1], full_t.iloc[len(train) - 1:]


class LogDiffWrapper(BaseForecaster):
    inner_module, inner_class, name = None, None, "logdiff"

    def __init__(self, inner):
        self.inner = inner

    def predict(self, hist, future_exog):
        r = np.asarray(self.inner.predict(to_logdiff(hist), future_exog), float)
        return np.expm1(np.log1p(hist["sales"].iloc[-1]) + np.cumsum(r))

    def describe(self):
        return {**self.inner.describe(), "target": "log1p_diff"}

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.inner.save(directory / "inner")
        write_json(directory / "meta.json", {"model": self.name, "target": "log1p_diff"})

    @classmethod
    def load(cls, directory: Path):
        inner_cls = getattr(importlib.import_module(cls.inner_module), cls.inner_class)
        return cls(inner_cls.load(directory / "inner"))


class XGBoostLogDiff(LogDiffWrapper):
    inner_module, inner_class, name = "src.models.xgboost_model", "XGBoostForecaster", "xgboost_logdiff"


class LSTMLogDiff(LogDiffWrapper):
    inner_module, inner_class, name = "src.models.lstm_model", "LSTMForecaster", "lstm_logdiff"

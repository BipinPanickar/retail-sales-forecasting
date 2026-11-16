"""Automated leakage checks (see docs/LEAKAGE_AUDIT.md)."""
import numpy as np
import pandas as pd

from src.evaluation.backtest import rolling_origin, chronological_split
from src.models.base import BaseForecaster


class Spy(BaseForecaster):
    """Records what each forecast call is allowed to see."""
    name = "spy"

    def __init__(self): self.calls = []

    def predict(self, hist, future_exog):
        self.calls.append((hist.index.max(), future_exog.index.min(), len(hist)))
        return np.zeros(len(future_exog))


def _df(n=60):
    return pd.DataFrame({"sales": np.arange(n, dtype=float)}, index=pd.date_range("2015-01-01", periods=n, freq="MS"))


def test_history_always_ends_before_forecast_window():
    spy, df = Spy(), _df()
    rolling_origin(spy, df, 40, 60, 6)
    assert spy.calls and all(h < f for h, f, _ in spy.calls)


def test_history_never_includes_test_actuals_beyond_origin():
    spy, df = Spy(), _df()
    rolling_origin(spy, df, 40, 60, 6)
    assert [n for _, _, n in spy.calls] == [40, 46, 52, 58]


def test_split_order_train_val_test():
    tr, va, te = chronological_split(_df(100), 0.7, 0.15)
    assert tr.index.max() < va.index.min() and va.index.max() < te.index.min()

import numpy as np
import pandas as pd

from src.analysis.tsa import diff_series, stationarity_verdict, rolling_stats


def test_diff_series_lengths():
    s = pd.Series(np.arange(40, dtype=float))
    assert len(diff_series(s, d=1)) == 39
    assert len(diff_series(s, d=1, D=1, m=12)) == 27


def test_verdict_logic():
    assert stationarity_verdict(0.01, 0.10).startswith("Stationary")
    assert stationarity_verdict(0.90, 0.01).startswith("Non-stationary")
    assert stationarity_verdict(0.01, 0.01).startswith("Conflicting")


def test_rolling_stats_columns():
    out = rolling_stats(pd.Series(np.arange(30.0)), [3])
    assert {"mean_3", "std_3"} <= set(out.columns)

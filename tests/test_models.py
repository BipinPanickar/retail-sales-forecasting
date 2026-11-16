import numpy as np
import pandas as pd

from src.evaluation.metrics import mae, rmse, smape, wape, mape
from src.evaluation.backtest import chronological_split, rolling_origin
from src.models.baseline import NaiveForecaster, SeasonalNaiveForecaster


def _df(n=48):
    idx = pd.date_range("2018-01-01", periods=n, freq="MS")
    return pd.DataFrame({"sales": 100 + 10 * np.sin(np.arange(n) * 2 * np.pi / 12)}, index=idx)


def test_metrics_known_values():
    assert mae([1, 2], [2, 4]) == 1.5
    assert abs(rmse([0, 0], [3, 4]) - np.sqrt(12.5)) < 1e-9
    assert wape([10, 10], [8, 12]) == 20.0
    assert np.isnan(mape([0, 0], [1, 1]))                       # zeros are not divided by


def test_split_is_chronological_and_disjoint():
    df = _df()
    tr, va, te = chronological_split(df, 0.7, 0.15)
    assert tr.index.max() < va.index.min() < va.index.max() < te.index.min()
    assert len(tr) + len(va) + len(te) == len(df)


def test_seasonal_naive_is_perfect_on_pure_seasonality():
    df = _df()
    r = rolling_origin(SeasonalNaiveForecaster(12), df, 36, 48, 6)
    assert mae(r["actual"], r["pred"]) < 1e-9


def test_rolling_origin_windows_cover_test_once():
    df = _df()
    r = rolling_origin(NaiveForecaster(), df, 36, 48, 5)       # 5,5,2
    assert list(r["date"]) == list(df.index[36:48])
    assert r["step"].max() == 5

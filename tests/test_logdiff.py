import numpy as np
import pandas as pd

from src.models.base import BaseForecaster
from src.models.logdiff import LogDiffWrapper, to_logdiff, split_logdiff


class ConstReturn(BaseForecaster):
    """Dummy inner model: always predicts a constant log-return r."""
    name = "const"

    def __init__(self, r): self.r = r
    def predict(self, hist, future_exog): return np.full(len(future_exog), self.r)
    def describe(self): return {}


def _df(vals):
    return pd.DataFrame({"sales": vals}, index=pd.date_range("2020-01-01", periods=len(vals), freq="MS"))


def test_logdiff_roundtrip_growth():
    hist = _df([100.0, 110.0, 121.0])
    r = np.log1p(121.0 * 1.1) - np.log1p(121.0)                     # constant growth step
    pred = LogDiffWrapper(ConstReturn(r)).predict(hist, pd.DataFrame(index=pd.date_range("2020-04-01", periods=1, freq="MS")))
    assert abs(pred[0] - 121.0 * 1.1) < 1e-6


def test_logdiff_handles_zero_sales():
    out = to_logdiff(_df([100.0, 0.0, 50.0]))                       # log1p keeps zero finite
    assert np.isfinite(out["sales"]).all()


def test_split_logdiff_keeps_first_validation_diff():
    tr, va = _df([1, 2, 3, 4.0]), _df([5, 6.0])
    va.index = pd.date_range("2020-05-01", periods=2, freq="MS")
    t, v = split_logdiff(tr, va)
    assert len(t) == 3 and len(v) == 2 and v.index[0] == va.index[0]

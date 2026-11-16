"""ARIMA (and the shared SARIMAX machinery reused by SARIMA).

Order selection (all on training/validation data only, never on test):
  1. d from the ADF test on TRAIN (p < 0.05 -> stationary -> d=0, else d=1).
  2. Fit every (p,q) in the grid on TRAIN, rank by AIC.
  3. Take the top-3 by AIC and pick the one with the lowest rolling-origin MAE on VALIDATION.
Parameters are stored as JSON; at forecast time they are re-applied to the latest history
with SARIMAX.smooth(), so no re-estimation and no pickle files are needed.
"""
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.backtest import rolling_origin
from src.evaluation.metrics import mae
from src.models.base import BaseForecaster, write_json, read_json
from src.utils.logger import get_logger

log = get_logger("sarimax")


def choose_d(y: pd.Series, alpha: float = 0.05) -> int:
    from src.analysis.tsa import adf_test
    p = adf_test(y)["p"]
    log.info("ADF p-value on train = %.4f -> d=%d", p, 0 if p < alpha else 1)
    return 0 if p < alpha else 1


class SarimaxForecaster(BaseForecaster):
    name = "sarimax"

    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), params=None, maxiter=100):
        self.order, self.seasonal_order, self.params_, self.maxiter = tuple(order), tuple(seasonal_order), params, maxiter
        self.aic_ = None

    def _trend(self):
        return "c" if (self.order[1] == 0 and self.seasonal_order[1] == 0) else None

    def _model(self, y):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        return SARIMAX(np.asarray(y, float), order=self.order, seasonal_order=self.seasonal_order,
                       trend=self._trend(), enforce_stationarity=False, enforce_invertibility=False)

    def fit_on(self, train: pd.DataFrame):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._model(train["sales"].values).fit(disp=False, maxiter=self.maxiter)
        self.params_, self.aic_ = np.asarray(res.params), float(res.aic)
        return self

    def predict(self, hist, future_exog):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._model(hist["sales"].values).smooth(self.params_)
            return np.asarray(res.forecast(len(future_exog)), float)

    def describe(self):
        return {"order": self.order, "seasonal_order": self.seasonal_order, "aic_train": self.aic_}

    def save(self, directory: Path):
        write_json(directory / "meta.json", {
            "model": self.name, "order": self.order, "seasonal_order": self.seasonal_order,
            "params": [float(x) for x in self.params_], "aic_train": self.aic_})

    @classmethod
    def load(cls, directory: Path):
        m = read_json(directory / "meta.json")
        obj = cls(m["order"], m["seasonal_order"], np.array(m["params"]))
        obj.name, obj.aic_ = m["model"], m.get("aic_train")
        return obj


def select_by_aic_then_validation(candidates, train, val, horizon, top_k=3, name="sarimax"):
    """candidates: iterable of (order, seasonal_order). Returns best fitted SarimaxForecaster."""
    fitted = []
    for order, sorder in candidates:
        try:
            m = SarimaxForecaster(order, sorder).fit_on(train)
            fitted.append(m)
        except Exception as e:  # numerical failures are expected for some orders
            log.warning("skip %s %s: %s", order, sorder, e)
    if not fitted:
        raise RuntimeError("No SARIMAX candidate could be fitted.")
    fitted.sort(key=lambda m: m.aic_)
    full = pd.concat([train, val])
    best, best_mae = None, np.inf
    for m in fitted[:top_k]:
        r = rolling_origin(m, full, len(train), len(full), horizon)
        score = mae(r["actual"], r["pred"])
        log.info("%s %s %s: AIC=%.1f val MAE=%.2f", name, m.order, m.seasonal_order, m.aic_, score)
        if score < best_mae:
            best, best_mae = m, score
    best.name = name
    return best


class ArimaForecaster(SarimaxForecaster):
    name = "arima"

    @staticmethod
    def select(train, val, horizon, cfg_arima, quick=False):
        d = choose_d(train["sales"])
        pmax, qmax = (1, 1) if quick else (cfg_arima["max_p"], cfg_arima["max_q"])
        cands = [((p, d, q), (0, 0, 0, 0)) for p, q in itertools.product(range(pmax + 1), range(qmax + 1))
                 if (p, q) != (0, 0) or d > 0]
        return select_by_aic_then_validation(cands, train, val, horizon, name="arima")

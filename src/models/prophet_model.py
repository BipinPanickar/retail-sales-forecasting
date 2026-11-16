"""Prophet (formerly fbprophet; the package is now called `prophet`).

ADDED AT THE USER'S REQUEST - it was not in the original brief's model list.
Important behavioural difference: Prophet fits trend + seasonality curves to the TRAINING
period and extrapolates them. It is NOT conditioned on recent actuals the way ARIMA/SARIMA/
XGBoost/LSTM are, so over a long test period its trend can drift. That is a property of the
method and is reported as-is.
"""
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.backtest import rolling_origin, get_exog_cols
from src.evaluation.metrics import mae
from src.models.base import BaseForecaster, write_json, read_json
from src.utils.logger import get_logger

log = get_logger("prophet_model")
GRID = {"changepoint_prior_scale": [0.05, 0.5], "seasonality_mode": ["additive", "multiplicative"]}


class ProphetForecaster(BaseForecaster):
    name = "prophet"

    def __init__(self, freq, exog_cols=(), params=None):
        self.freq, self.exog_cols, self.params = freq, list(exog_cols), params or {}
        self.model = None

    def _new(self):
        for n in ("cmdstanpy", "prophet"):
            logging.getLogger(n).setLevel(logging.WARNING)
        from prophet import Prophet
        monthly = self.freq.upper().startswith("M")
        m = Prophet(yearly_seasonality=True, weekly_seasonality=not monthly,
                    daily_seasonality=False, **self.params)
        for c in self.exog_cols:
            m.add_regressor(c)
        return m

    def fit_on(self, train: pd.DataFrame):
        df = pd.DataFrame({"ds": train.index, "y": train["sales"].values})
        for c in self.exog_cols:
            df[c] = train[c].values
        self.model = self._new()
        self.model.fit(df)
        return self

    def predict(self, hist, future_exog):
        fut = pd.DataFrame({"ds": future_exog.index})
        for c in self.exog_cols:
            fut[c] = future_exog[c].values
        return self.model.predict(fut)["yhat"].values.astype(float)

    def describe(self):
        return dict(self.params)

    def save(self, directory: Path):
        from prophet.serialize import model_to_json
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.json").write_text(model_to_json(self.model), encoding="utf-8")
        write_json(directory / "meta.json", {"model": self.name, "freq": self.freq,
                                             "exog_cols": self.exog_cols, "params": self.params})

    @classmethod
    def load(cls, directory: Path):
        from prophet.serialize import model_from_json
        m = read_json(directory / "meta.json")
        obj = cls(m["freq"], m["exog_cols"], m["params"])
        obj.model = model_from_json((directory / "model.json").read_text(encoding="utf-8"))
        return obj

    @staticmethod
    def select(train, val, horizon, dcfg, quick=False):
        exog = get_exog_cols(train)
        grid = {k: v[:1] for k, v in GRID.items()} if quick else GRID
        keys, full = list(grid), pd.concat([train, val])
        best, best_mae = None, np.inf
        for vals in itertools.product(*[grid[k] for k in keys]):
            params = dict(zip(keys, vals))
            m = ProphetForecaster(dcfg["freq"], exog, params).fit_on(train)
            r = rolling_origin(m, full, len(train), len(full), horizon)
            score = mae(r["actual"], r["pred"])
            log.info("prophet %s -> val MAE %.3f", params, score)
            if score < best_mae:
                best, best_mae = m, score
        return best

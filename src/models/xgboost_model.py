"""XGBoost on lag / rolling / calendar features with recursive multi-step forecasting.

Known limitation (report honestly): tree models cannot extrapolate beyond the target range
seen in training, which matters for trending series such as US retail sales.
"""
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.backtest import rolling_origin, get_exog_cols
from src.evaluation.metrics import mae
from src.features.feature_engineering import make_features, feature_row
from src.models.base import BaseForecaster, write_json, read_json
from src.utils.logger import get_logger

log = get_logger("xgboost")


class XGBoostForecaster(BaseForecaster):
    name = "xgboost"

    def __init__(self, lags, windows, freq, exog_cols=(), params=None, seed=42):
        self.lags, self.windows, self.freq = list(lags), list(windows), freq
        self.exog_cols, self.params, self.seed = list(exog_cols), params or {}, seed
        self.model, self.columns = None, None

    def fit_on(self, train: pd.DataFrame):
        import xgboost as xgb
        X = make_features(train, self.lags, self.windows, self.exog_cols, self.freq)
        ok = X.notna().all(axis=1)
        self.columns = list(X.columns)
        self.model = xgb.XGBRegressor(random_state=self.seed, n_jobs=-1, objective="reg:squarederror", **self.params)
        self.model.fit(X[ok], train.loc[ok, "sales"])
        return self

    def predict(self, hist, future_exog):
        y = list(hist["sales"].values)
        out = []
        for i, dt in enumerate(future_exog.index):
            row = feature_row(y, dt, future_exog.iloc[i], self.lags, self.windows,
                              self.exog_cols, self.freq, self.columns)
            yhat = float(self.model.predict(row)[0])
            out.append(yhat)
            y.append(yhat)  # recursive forecasting
        return np.array(out)

    def describe(self):
        return {**self.params, "lags": self.lags, "windows": self.windows}

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=self.columns).sort_values(ascending=False)

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(directory / "model.json"))
        write_json(directory / "meta.json", {
            "model": self.name, "lags": self.lags, "windows": self.windows, "freq": self.freq,
            "exog_cols": self.exog_cols, "params": self.params, "columns": self.columns, "seed": self.seed})

    @classmethod
    def load(cls, directory: Path):
        import xgboost as xgb
        m = read_json(directory / "meta.json")
        obj = cls(m["lags"], m["windows"], m["freq"], m["exog_cols"], m["params"], m["seed"])
        obj.columns = m["columns"]
        obj.model = xgb.XGBRegressor()
        obj.model.load_model(str(directory / "model.json"))
        return obj

    @staticmethod
    def select(train, val, horizon, dcfg, grid, seed=42, quick=False, logdiff=False):
        exog = get_exog_cols(train)
        if logdiff:
            from src.models.logdiff import split_logdiff, XGBoostLogDiff
            train_t, _ = split_logdiff(train, val)
        else:
            train_t = train
        if quick:
            grid = {k: v[:1] for k, v in grid.items()}
        keys = list(grid)
        full = pd.concat([train, val])
        best, best_mae = None, np.inf
        for vals in itertools.product(*[grid[k] for k in keys]):
            params = dict(zip(keys, vals))
            m = XGBoostForecaster(dcfg["lags"], dcfg["rolling_windows"], dcfg["freq"], exog, params, seed).fit_on(train_t)
            m = XGBoostLogDiff(m) if logdiff else m
            r = rolling_origin(m, full, len(train), len(full), horizon)
            score = mae(r["actual"], r["pred"])
            log.info("xgb %s -> val MAE %.3f", params, score)
            if score < best_mae:
                best, best_mae = m, score
        return best

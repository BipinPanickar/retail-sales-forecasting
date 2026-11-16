"""Naive and seasonal-naive baselines. Any complex model must beat these."""
import numpy as np

from src.models.base import BaseForecaster, write_json, read_json


class NaiveForecaster(BaseForecaster):
    """Forecast(t+h) = Actual(t) (last observed value, repeated)."""
    name = "naive"

    def fit(self, train, val=None):
        return self

    def predict(self, hist, future_exog):
        return np.repeat(hist["sales"].iloc[-1], len(future_exog)).astype(float)

    def save(self, directory):
        write_json(directory / "meta.json", {"model": self.name})

    @classmethod
    def load(cls, directory):
        return cls()


class SeasonalNaiveForecaster(BaseForecaster):
    """Forecast(t) = Actual(t - season_length); cycles if horizon > season_length."""
    name = "seasonal_naive"

    def __init__(self, season_length: int):
        self.m = season_length

    def fit(self, train, val=None):
        return self

    def predict(self, hist, future_exog):
        v = hist["sales"].values
        h = len(future_exog)
        return v[len(v) - self.m + (np.arange(h) % self.m)].astype(float)

    def describe(self):
        return {"season_length": self.m}

    def save(self, directory):
        write_json(directory / "meta.json", {"model": self.name, "season_length": self.m})

    @classmethod
    def load(cls, directory):
        return cls(read_json(directory / "meta.json")["season_length"])

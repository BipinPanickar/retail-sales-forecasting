"""Common interface so every model is evaluated by exactly the same code."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


class BaseForecaster:
    name = "base"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame = None):
        raise NotImplementedError

    def predict(self, hist: pd.DataFrame, future_exog: pd.DataFrame) -> np.ndarray:
        """Forecast len(future_exog) steps ahead from the END of `hist` (no refitting)."""
        raise NotImplementedError

    # persistence ---------------------------------------------------------
    def save(self, directory: Path):
        raise NotImplementedError

    @classmethod
    def load(cls, directory: Path):
        raise NotImplementedError

    def describe(self) -> dict:
        return {}


def write_json(path: Path, obj: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

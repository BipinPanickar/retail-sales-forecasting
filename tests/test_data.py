import sqlite3
import numpy as np
import pandas as pd

from src.data.ingestion import SCHEMA, store_observations
from src.data.preprocessing import clean_series
from src.data.validation import validate_fred_observations, ValidationError


def _conn():
    c = sqlite3.connect(":memory:")
    c.executescript(SCHEMA)
    return c


def test_fred_missing_dot_becomes_nan_not_zero():
    df = validate_fred_observations([{"date": "2020-01-01", "value": "."}, {"date": "2020-02-01", "value": "10"}])
    assert np.isnan(df["value"].iloc[0]) and df["value"].iloc[1] == 10


def test_negative_sales_rejected():
    try:
        validate_fred_observations([{"date": "2020-01-01", "value": "-5"}])
        assert False, "should raise"
    except ValidationError:
        pass


def test_ingestion_is_idempotent_and_keeps_revisions():
    c = _conn()
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-02-01"]), "value": [100.0, 110.0]})
    assert store_observations(c, "X", df) == (2, 0)
    assert store_observations(c, "X", df) == (0, 0)            # re-run: no duplicates
    rev = df.copy(); rev.loc[1, "value"] = 112.0
    assert store_observations(c, "X", rev) == (0, 1)           # revision stored as new version
    assert c.execute("SELECT COUNT(*) FROM raw_sales").fetchone()[0] == 3


def test_clean_series_fills_gap_and_flags():
    idx = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-04-01"])
    out = clean_series(pd.Series([1.0, 2.0, 4.0], index=idx), "MS")
    assert len(out) == 4 and out["imputed"].sum() == 1 and out["sales"].iloc[2] == 3.0

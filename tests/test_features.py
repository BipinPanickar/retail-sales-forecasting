import numpy as np
import pandas as pd

from src.features.feature_engineering import make_features, feature_row


def _df(n=60):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"sales": np.arange(n, dtype=float) ** 1.1 + 5}, index=idx)


def test_no_future_leakage_in_features():
    df = _df()
    X1 = make_features(df, [1, 7], [7], [], "D")
    df2 = df.copy(); df2.iloc[40:, 0] = 9999.0                 # change the future
    X2 = make_features(df2, [1, 7], [7], [], "D")
    pd.testing.assert_frame_equal(X1.iloc[:40], X2.iloc[:40])  # rows before t=40 unchanged


def test_row_builder_matches_vectorised_features():
    df = _df()
    X = make_features(df, [1, 7], [7], [], "D")
    t = 30
    row = feature_row(df["sales"].iloc[:t].values, df.index[t], pd.Series(dtype=float),
                      [1, 7], [7], [], "D", list(X.columns))
    np.testing.assert_allclose(row.iloc[0].values.astype(float), X.iloc[t].values.astype(float))

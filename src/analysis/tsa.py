"""Time-series analysis helpers: pure computation, no plotting (the dashboard plots them).

Methodological note: ACF/PACF/stationarity results are used to CHOOSE model orders, so they
should be computed on TRAINING data only; using the full series lets test-period information
influence model identification (mild leakage). The dashboard defaults to train-only.
"""
import warnings

import numpy as np
import pandas as pd


def rolling_stats(s: pd.Series, windows) -> pd.DataFrame:
    out = pd.DataFrame({"sales": s})
    for w in windows:
        out[f"mean_{w}"] = s.rolling(w).mean()
        out[f"std_{w}"] = s.rolling(w).std()
    return out


def stl_decompose(s: pd.Series, period: int, robust: bool = True) -> pd.DataFrame:
    from statsmodels.tsa.seasonal import STL
    r = STL(s, period=period, robust=robust).fit()
    return pd.DataFrame({"observed": s, "trend": r.trend, "seasonal": r.seasonal, "resid": r.resid})


def classical_decompose(s: pd.Series, period: int, model: str = "additive") -> pd.DataFrame:
    from statsmodels.tsa.seasonal import seasonal_decompose
    r = seasonal_decompose(s, model=model, period=period)
    return pd.DataFrame({"observed": r.observed, "trend": r.trend, "seasonal": r.seasonal, "resid": r.resid})


def component_strength(dec: pd.DataFrame) -> dict:
    """Trend / seasonal strength in [0,1] (Hyndman & Athanasopoulos): 1 - Var(R)/Var(C+R).
    Meaningful for additive decompositions (STL, additive classical)."""
    r = dec["resid"].dropna()
    t, se = dec["trend"].reindex(r.index), dec["seasonal"].reindex(r.index)
    return {"trend_strength": float(max(0.0, 1 - r.var() / (t + r).var())),
            "seasonal_strength": float(max(0.0, 1 - r.var() / (se + r).var()))}


def acf_pacf(s: pd.Series, nlags: int):
    """Returns (DataFrame[lag, acf, pacf], band). band = approximate 95% white-noise limit."""
    from statsmodels.tsa.stattools import acf, pacf
    x = s.dropna().values.astype(float)
    n = len(x)
    nlags = int(min(nlags, n // 2 - 1))
    a = acf(x, nlags=nlags, fft=True)
    p = pacf(x, nlags=nlags, method="ywm")
    return pd.DataFrame({"lag": np.arange(nlags + 1), "acf": a, "pacf": p}), float(1.96 / np.sqrt(n))


def adf_test(s: pd.Series) -> dict:
    """H0: unit root (non-stationary). Small p-value -> reject H0 -> stationary."""
    from statsmodels.tsa.stattools import adfuller
    x = s.dropna().values.astype(float)
    try:
        res = adfuller(x, autolag="AIC", result_object=False)
    except TypeError:                      # older statsmodels without result_object
        res = adfuller(x, autolag="AIC")
    return {"stat": float(res[0]), "p": float(res[1]), "crit_5pct": float(res[4]["5%"])}


def kpss_test(s: pd.Series, regression: str = "c") -> dict:
    """H0: stationary around a constant ('c') or trend ('ct'). Small p -> reject -> NON-stationary.
    statsmodels interpolates p within [0.01, 0.10]."""
    from statsmodels.tsa.stattools import kpss
    x = s.dropna().values.astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, _, crit = kpss(x, regression=regression, nlags="auto")
    return {"stat": float(stat), "p": float(p), "crit_5pct": float(crit["5%"])}


def stationarity_verdict(adf_p: float, kpss_p: float, alpha: float = 0.05) -> str:
    adf_stationary = adf_p < alpha
    kpss_stationary = kpss_p >= alpha
    if adf_stationary and kpss_stationary:
        return "Stationary (both tests agree)"
    if not adf_stationary and not kpss_stationary:
        return "Non-stationary (both tests agree)"
    if adf_stationary:
        return "Conflicting: ADF says stationary, KPSS says not (trend-stationary or structural change?)"
    return "Conflicting: ADF says non-stationary, KPSS says stationary (low test power or structural break?)"


def diff_series(s: pd.Series, d: int = 0, D: int = 0, m: int = 12) -> pd.Series:
    """Apply D seasonal differences (lag m) and d regular differences."""
    out = s.copy()
    for _ in range(D):
        out = out.diff(m)
    for _ in range(d):
        out = out.diff()
    return out.dropna()


def ljung_box_p(resid, lag: int) -> float:
    """H0: no autocorrelation up to `lag`. Small p -> leftover autocorrelation."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    r = pd.Series(np.asarray(resid, float)).dropna()
    lag = int(min(lag, len(r) // 2 - 1))
    return float(acorr_ljungbox(r, lags=[lag], return_df=True)["lb_pvalue"].iloc[0])

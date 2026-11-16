"""Forecast accuracy metrics. MAPE is guarded against zeros; sMAPE and WAPE are provided
because retail data can contain values near zero."""
import numpy as np


def mae(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mape(y, p):
    """Mean Absolute Percentage Error (%), zeros in y are excluded (and counted nowhere else)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y != 0
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100) if m.any() else float("nan")


def smape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    denom = (np.abs(y) + np.abs(p)) / 2
    m = denom != 0
    return float(np.mean(np.abs(y[m] - p[m]) / denom[m]) * 100) if m.any() else float("nan")


def wape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    s = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - p)) / s * 100) if s else float("nan")


def all_metrics(y, p) -> dict:
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE": mape(y, p),
            "sMAPE": smape(y, p), "WAPE": wape(y, p)}

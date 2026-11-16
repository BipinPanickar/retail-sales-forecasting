"""Forecasting engine: load a saved model and forecast from the END of the latest data.

NOTE: saved models were fitted on the TRAIN period only. At forecast time they are
conditioned on the full latest history (no re-estimation). A scheduled retraining job is
a Phase-11 decision (see README, 'Retraining strategy').
"""
import argparse
from datetime import datetime, timezone

import pandas as pd

from src.data.ingestion import get_connection
from src.data.preprocessing import load_fred_series, clean_series
from src.models.baseline import NaiveForecaster, SeasonalNaiveForecaster
from src.utils.config import load_config, PROJECT_ROOT

MODEL_CLASSES = {
    "naive": ("src.models.baseline", "NaiveForecaster"),
    "seasonal_naive": ("src.models.baseline", "SeasonalNaiveForecaster"),
    "arima": ("src.models.arima_model", "ArimaForecaster"),
    "sarima": ("src.models.sarima_model", "SarimaForecaster"),
    "xgboost": ("src.models.xgboost_model", "XGBoostForecaster"),
    "lstm": ("src.models.lstm_model", "LSTMForecaster"),
    "prophet": ("src.models.prophet_model", "ProphetForecaster"),
    "xgboost_logdiff": ("src.models.logdiff", "XGBoostLogDiff"),
    "lstm_logdiff": ("src.models.logdiff", "LSTMLogDiff"),
}


def model_dir(cfg, model_name, series_id):
    sub = {"seasonal_naive": "baseline", "naive": "baseline"}.get(model_name, model_name)
    return PROJECT_ROOT / cfg["paths"]["models_dir"] / sub / f"fred_monthly_{series_id}_{model_name}"


def load_model(cfg, model_name, series_id):
    import importlib
    mod, cls = MODEL_CLASSES[model_name]
    return getattr(importlib.import_module(mod), cls).load(model_dir(cfg, model_name, series_id))


def generate_forecast(series_id: str, model_name: str, horizon: int, save: bool = False) -> pd.DataFrame:
    cfg = load_config()
    freq = cfg["datasets"]["fred_monthly"]["freq"]
    hist = clean_series(load_fred_series(series_id), freq, series_id)
    model = load_model(cfg, model_name, series_id)
    idx = pd.date_range(hist.index[-1], periods=horizon + 1, freq=freq)[1:]
    pred = model.predict(hist, pd.DataFrame(index=idx))
    out = pd.DataFrame({"forecast_date": idx, "model": model_name, "predicted_sales": pred})
    if save:
        conn = get_connection(cfg)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        conn.executemany("INSERT INTO forecasts VALUES (?,?,?,?,?)",
                         [(d.strftime("%Y-%m-%d"), series_id, model_name, float(p), now)
                          for d, p in zip(idx, pred)])
        conn.commit(); conn.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="RSAFSNA")
    ap.add_argument("--model", default="sarima", choices=list(MODEL_CLASSES))
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    print(generate_forecast(a.series, a.model, a.horizon, a.save).to_string(index=False))

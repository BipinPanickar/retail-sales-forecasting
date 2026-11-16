"""Run the full model comparison for ONE dataset with the same chronological split.

Usage:
  python -m src.run_experiments --dataset fred_monthly --series RSAFSNA
  python -m src.run_experiments --dataset m5_daily
  add --quick for a fast smoke test, --models naive,seasonal_naive,arima,sarima,xgboost,lstm

Outputs: results/experiments.csv (one row per model per split), results/<tag>_test_predictions.csv,
saved models under models/.
"""
import argparse
import time
import uuid
from datetime import datetime, timezone

import pandas as pd

from src.data.preprocessing import clean_series, load_fred_series
from src.evaluation.backtest import chronological_split, rolling_origin, get_exog_cols
from src.evaluation.metrics import all_metrics
from src.models.baseline import NaiveForecaster, SeasonalNaiveForecaster
from src.utils.config import load_config, set_seed, resolve, PROJECT_ROOT
from src.utils.logger import get_logger

log = get_logger("experiments")
ALL = ["naive", "seasonal_naive", "arima", "sarima", "prophet", "xgboost", "xgboost_logdiff", "lstm", "lstm_logdiff"]


def load_dataset(cfg, dataset, series_id):
    if dataset == "fred_monthly":
        return clean_series(load_fred_series(series_id), cfg["datasets"][dataset]["freq"], series_id)
    if dataset == "m5_daily":
        from src.data.m5_loader import load_m5_daily
        df = load_m5_daily()
        return df.asfreq("D")
    raise ValueError(dataset)


def build_and_fit(name, cfg, dcfg, dataset, train, val, horizon, quick, tag):
    seed = cfg["project"]["seed"]
    if name == "naive":
        return NaiveForecaster().fit(train)
    if name == "seasonal_naive":
        return SeasonalNaiveForecaster(dcfg["season_length"]).fit(train)
    if name == "arima":
        from src.models.arima_model import ArimaForecaster
        return ArimaForecaster.select(train, val, horizon, cfg["arima"], quick)
    if name == "sarima":
        from src.models.sarima_model import SarimaForecaster
        return SarimaForecaster.select(train, val, horizon, dcfg["season_length"], cfg["sarima"], quick)
    if name == "prophet":
        from src.models.prophet_model import ProphetForecaster
        return ProphetForecaster.select(train, val, horizon, dcfg, quick)
    if name == "xgboost":
        from src.models.xgboost_model import XGBoostForecaster
        return XGBoostForecaster.select(train, val, horizon, dcfg, cfg["xgboost"]["param_grid"], seed, quick)
    if name == "xgboost_logdiff":
        from src.models.xgboost_model import XGBoostForecaster
        from src.models.logdiff import XGBoostLogDiff
        return XGBoostForecaster.select(train, val, horizon, dcfg, cfg["xgboost"]["param_grid"], seed, quick, logdiff=True)
    if name == "lstm":
        from src.models.lstm_model import LSTMForecaster
        ck = PROJECT_ROOT / cfg["paths"]["models_dir"] / "lstm"
        return LSTMForecaster.select(train, val, horizon, dcfg, cfg["lstm"], seed, quick, ck)
    if name == "lstm_logdiff":
        from src.models.lstm_model import LSTMForecaster
        ck = PROJECT_ROOT / cfg["paths"]["models_dir"] / "lstm"
        return LSTMForecaster.select(train, val, horizon, dcfg, cfg["lstm"], seed, quick, ck, logdiff=True)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fred_monthly", choices=["fred_monthly", "m5_daily"])
    ap.add_argument("--series", default="RSAFSNA")
    ap.add_argument("--models", default=",".join(ALL))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    cfg = load_config()
    seed = cfg["project"]["seed"]
    set_seed(seed)
    dcfg = cfg["datasets"][a.dataset]
    horizon = dcfg["horizon"]
    series = a.series if a.dataset == "fred_monthly" else "total"
    tag = f"{a.dataset}_{series}"

    df = load_dataset(cfg, a.dataset, a.series)
    train, val, test = chronological_split(df, cfg["split"]["train_frac"], cfg["split"]["val_frac"])
    n_tr, n_va = len(train), len(val)
    log.info("%s: train=%d val=%d test=%d | horizon=%d", tag, n_tr, n_va, len(test), horizon)

    exp_id, rows, preds = uuid.uuid4().hex[:8], [], []
    for name in [m.strip() for m in a.models.split(",")]:
        log.info("=== %s ===", name)
        t0 = time.time()
        try:
            model = build_and_fit(name, cfg, dcfg, a.dataset, train, val, horizon, a.quick, tag)
        except ImportError as e:
            log.warning("Skipping %s: package missing (%s)", name, e)
            continue
        train_time = time.time() - t0
        for split, s, e in (("validation", n_tr, n_tr + n_va), ("test", n_tr + n_va, len(df))):
            r = rolling_origin(model, df, s, e, horizon)
            met = all_metrics(r["actual"], r["pred"])
            rows.append({"experiment_id": exp_id, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "dataset": a.dataset, "series": series, "model": name, "split": split,
                         "parameters": str(model.describe()), "horizon": horizon, "seed": seed,
                         "training_start": train.index[0].date(), "training_end": train.index[-1].date(),
                         "validation_start": val.index[0].date(), "validation_end": val.index[-1].date(),
                         "test_start": test.index[0].date(), "test_end": test.index[-1].date(),
                         **met, "training_time_s": round(train_time, 2),
                         "notes": "quick" if a.quick else ""})
            if split == "test":
                preds.append(r.assign(model=name))
        sub = {"naive": "baseline", "seasonal_naive": "baseline"}.get(name, name)
        model.save(PROJECT_ROOT / cfg["paths"]["models_dir"] / sub / f"{tag}_{name}")

    res_path = resolve(f"{cfg['paths']['results_dir']}/experiments.csv")
    new = pd.DataFrame(rows)
    new.to_csv(res_path, mode="a", header=not res_path.exists(), index=False)
    pd.concat(preds).to_csv(resolve(f"{cfg['paths']['results_dir']}/{tag}_test_predictions.csv"), index=False)
    show = new[new["split"] == "test"][["model", "MAE", "RMSE", "sMAPE", "WAPE", "training_time_s"]]
    print("\nTEST-SET RESULTS (measured, experiment", exp_id, ")\n", show.to_string(index=False))


if __name__ == "__main__":
    main()

"""Repeat the stochastic models over several random seeds and report mean/std of test error.

Usage:
  python -m src.run_seeds --dataset fred_monthly --series RSAFSNA --models lstm,lstm_logdiff,xgboost_logdiff --seeds 5
Each seed re-runs the full validation-based tuning for that model (so it takes about
seeds x the single-run time). Output: results/<tag>_seeds.csv and a summary table.
ARIMA/SARIMA/baselines are deterministic and need no seeds.
"""
import argparse

import pandas as pd

from src.evaluation.backtest import chronological_split, rolling_origin
from src.evaluation.metrics import all_metrics
from src.run_experiments import build_and_fit, load_dataset
from src.utils.config import load_config, set_seed, resolve
from src.utils.logger import get_logger

log = get_logger("seeds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fred_monthly", choices=["fred_monthly", "m5_daily"])
    ap.add_argument("--series", default="RSAFSNA")
    ap.add_argument("--models", default="lstm,lstm_logdiff,xgboost_logdiff")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    cfg = load_config()
    dcfg = cfg["datasets"][a.dataset]
    series = a.series if a.dataset == "fred_monthly" else "total"
    tag = f"{a.dataset}_{series}"
    df = load_dataset(cfg, a.dataset, a.series)
    train, val, test = chronological_split(df, cfg["split"]["train_frac"], cfg["split"]["val_frac"])
    n_tr, n_va, H = len(train), len(val), dcfg["horizon"]
    rows = []
    for seed in range(cfg["project"]["seed"], cfg["project"]["seed"] + a.seeds):
        cfg["project"]["seed"] = seed
        set_seed(seed)
        for name in [m.strip() for m in a.models.split(",")]:
            log.info("seed=%d model=%s", seed, name)
            model = build_and_fit(name, cfg, dcfg, a.dataset, train, val, H, a.quick, tag)
            r = rolling_origin(model, df, n_tr + n_va, len(df), H)
            rows.append({"seed": seed, "model": name, **all_metrics(r["actual"], r["pred"])})
    res = pd.DataFrame(rows)
    res.to_csv(resolve(f"{cfg['paths']['results_dir']}/{tag}_seeds.csv"), index=False)
    summ = res.groupby("model")[["MAE", "RMSE", "WAPE"]].agg(["mean", "std", "min", "max"]).round(2)
    print("\nTEST metrics across seeds (measured):\n", summ.to_string())


if __name__ == "__main__":
    main()

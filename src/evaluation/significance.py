"""Diebold-Mariano tests on the test-period forecast errors (with the Harvey-Leybourne-Newbold
small-sample correction) and Holm adjustment across comparisons.

Usage:
  python -m src.evaluation.significance --dataset fred_monthly --series RSAFSNA --reference seasonal_naive
Output: results/<tag>_dm_tests.csv  (measured; nothing is hard-coded)

Interpretation: H0 = equal predictive accuracy. d_t = loss(model A) - loss(reference).
Negative mean difference = model A has LOWER loss than the reference. The forecasts are
multi-step blocks, so the DM variance uses h-1 autocovariances with h = the block horizon
(conservative). With ~60 test points, low power is expected: report effect sizes, not only p-values.
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config import load_config, PROJECT_ROOT


def dm_test(actual, pred_a, pred_b, h: int = 1, loss: str = "abs") -> dict:
    """Two-sided DM test. loss: 'abs' (MAE-type) or 'squared' (MSE-type)."""
    a, p1, p2 = (np.asarray(x, float) for x in (actual, pred_a, pred_b))
    e1, e2 = a - p1, a - p2
    d = (np.abs(e1) - np.abs(e2)) if loss == "abs" else (e1 ** 2 - e2 ** 2)
    n = len(d)
    dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    lrv = gamma0
    for k in range(1, h):
        lrv += 2 * np.mean((d[k:] - dbar) * (d[:-k] - dbar))
    if lrv <= 0 or n < 5:
        return {"mean_loss_diff": float(dbar), "dm_stat": np.nan, "p_value": np.nan, "n": n}
    dm = dbar / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)      # Harvey-Leybourne-Newbold
    stat = dm * hln
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return {"mean_loss_diff": float(dbar), "dm_stat": float(stat), "p_value": float(p), "n": n}


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    m = int(np.sum(~np.isnan(p)))
    adj = np.full(len(p), np.nan)
    running = 0.0
    for rank, idx in enumerate(order[:m]):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def run(dataset: str, series: str, reference: str, horizon: int) -> pd.DataFrame:
    cfg = load_config()
    tag = f"{dataset}_{'total' if dataset == 'm5_daily' else series}"
    path = PROJECT_ROOT / cfg["paths"]["results_dir"] / f"{tag}_test_predictions.csv"
    p = pd.read_csv(path, parse_dates=["date"])
    wide = p.pivot_table(index="date", columns="model", values="pred")
    actual = p.drop_duplicates("date").set_index("date").sort_index()["actual"]
    wide = wide.reindex(actual.index)
    if reference not in wide.columns:
        raise SystemExit(f"Reference model '{reference}' not in predictions: {list(wide.columns)}")
    rows = []
    for m in wide.columns:
        if m == reference:
            continue
        for loss in ("abs", "squared"):
            r = dm_test(actual.values, wide[m].values, wide[reference].values, h=horizon, loss=loss)
            rows.append({"model": m, "reference": reference, "loss": loss, **r})
    out = pd.DataFrame(rows)
    for loss in ("abs", "squared"):
        m_ = out["loss"] == loss
        out.loc[m_, "p_holm"] = holm(out.loc[m_, "p_value"].values)
    out["verdict_5pct"] = np.where(out["p_holm"] < 0.05,
                                   np.where(out["mean_loss_diff"] < 0, "better than reference", "worse than reference"),
                                   "no significant difference")
    out.to_csv(PROJECT_ROOT / cfg["paths"]["results_dir"] / f"{tag}_dm_tests.csv", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fred_monthly")
    ap.add_argument("--series", default="RSAFSNA")
    ap.add_argument("--reference", default="seasonal_naive")
    a = ap.parse_args()
    cfg = load_config()
    res = run(a.dataset, a.series, a.reference, cfg["datasets"][a.dataset]["horizon"])
    print(res.round(4).to_string(index=False))

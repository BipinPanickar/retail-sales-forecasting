"""Plots for the model comparison. Reads MEASURED results only.

Usage (after src.run_experiments has finished):
  python -m src.evaluation.plots --dataset fred_monthly --series RSAFSNA
  add --show to also open the windows interactively.

Creates PNG files in results/figures/:
  1_all_models_overlay.png   history + every model's test forecast on one chart
  2_per_model_panels.png     one panel per model, actual vs predicted
  3_error_by_step.png        MAE by forecast step (does accuracy change with horizon?)
  4_metrics_bar.png          test MAE and WAPE per model
  5_residuals.png            residuals (actual - predicted) over time per model

Forecasts are non-overlapping blocks of `horizon` steps, each started from actual history,
so each block is drawn as a separate line (no line joins one block to the next).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import load_config, PROJECT_ROOT

ORDER = ["naive", "seasonal_naive", "arima", "sarima", "prophet", "xgboost", "xgboost_logdiff", "lstm", "lstm_logdiff"]
COLORS = {"naive": "#9e9e9e", "seasonal_naive": "#8d6e63", "arima": "#1f77b4",
          "sarima": "#2ca02c", "prophet": "#9467bd", "xgboost_logdiff": "#e6a04a", "lstm_logdiff": "#8b1a1a", "xgboost": "#ff7f0e", "lstm": "#d62728"}


def _blocks(df):
    """Yield (origin, block) so each forecast block is drawn separately."""
    for origin, b in df.groupby("origin"):
        yield origin, b.sort_values("date")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fred_monthly")
    ap.add_argument("--series", default="RSAFSNA")
    ap.add_argument("--context_years", type=int, default=6, help="history shown before the test period")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    cfg = load_config()
    series = a.series if a.dataset == "fred_monthly" else "total"
    res_dir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    fig_dir = res_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(res_dir / f"{a.dataset}_{series}_test_predictions.csv", parse_dates=["date", "origin"])
    models = [m for m in ORDER if m in set(pred["model"])] + \
             [m for m in sorted(set(pred["model"])) if m not in ORDER]
    pred["resid"] = pred["actual"] - pred["pred"]

    proc = PROJECT_ROOT / cfg["paths"]["processed_dir"] / \
        (f"fred_{series}.csv" if a.dataset == "fred_monthly" else "m5_daily.csv")
    hist = pd.read_csv(proc, index_col=0, parse_dates=True)["sales"]
    test_start = pred["date"].min()
    ctx = hist[hist.index >= test_start - pd.DateOffset(years=a.context_years)]
    actual = pred[pred["model"] == models[0]].sort_values("date")

    # 1. overlay ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(ctx.index, ctx.values, color="black", lw=1.6, label="Actual")
    ax.axvline(test_start, color="grey", ls="--", lw=1)
    ax.text(test_start, ax.get_ylim()[1], "  test period starts", va="top", fontsize=9, color="grey")
    for m in models:
        first = True
        for _, b in _blocks(pred[pred["model"] == m]):
            ax.plot(b["date"], b["pred"], color=COLORS.get(m), lw=1.2, alpha=0.9,
                    label=m if first else None)
            first = False
    ax.set_title(f"Test-period forecasts, all models ({a.dataset}, {series})")
    ax.set_ylabel("Sales"); ax.legend(ncol=4, fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "1_all_models_overlay.png", dpi=150)

    # 2. per-model panels -------------------------------------------------
    n = len(models); cols = 2; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.6 * rows), sharex=True, sharey=True, squeeze=False)
    for ax, m in zip(axes.ravel(), models):
        d = pred[pred["model"] == m]
        ax.plot(actual["date"], actual["actual"], color="black", lw=1.4, label="Actual")
        for o, b in _blocks(d):
            ax.plot(b["date"], b["pred"], color=COLORS.get(m), lw=1.4)
            ax.axvline(o, color="lightgrey", lw=0.6)          # forecast origin
        mae = d["resid"].abs().mean()
        ax.set_title(f"{m}   (test MAE {mae:,.0f})", fontsize=11); ax.grid(alpha=0.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Actual (black) vs predicted; grey lines = forecast origins", y=1.0)
    fig.tight_layout(); fig.savefig(fig_dir / "2_per_model_panels.png", dpi=150)

    # 3. error by step ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m in models:
        g = pred[pred["model"] == m].assign(ae=lambda x: x["resid"].abs()).groupby("step")["ae"].mean()
        ax.plot(g.index, g.values, marker="o", color=COLORS.get(m), label=m)
    ax.set_xlabel("Forecast step (months/days ahead)"); ax.set_ylabel("Mean absolute error")
    ax.set_title("Does accuracy change with forecast horizon?"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "3_error_by_step.png", dpi=150)

    # 4. metrics bars (latest experiment for this dataset/series) ----------
    exp = pd.read_csv(res_dir / "experiments.csv")
    exp = exp[(exp["dataset"] == a.dataset) & (exp["series"] == series) & (exp["split"] == "test")]
    exp = exp[exp["experiment_id"] == exp.sort_values("timestamp")["experiment_id"].iloc[-1]]
    exp = exp.set_index("model").reindex([m for m in models if m in exp["model"].values])
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col in zip(axs, ["MAE", "WAPE"]):
        ax.bar(exp.index, exp[col], color=[COLORS.get(m) for m in exp.index])
        ax.set_title(f"Test {col}" + (" (%)" if col == "WAPE" else "")); ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(exp[col]):
            ax.text(i, v, f"{v:,.1f}", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    note = "quick run" if str(exp["notes"].iloc[0]) == "quick" else "full run"
    fig.suptitle(f"Experiment {exp['experiment_id'].iloc[0]} ({note})")
    fig.tight_layout(); fig.savefig(fig_dir / "4_metrics_bar.png", dpi=150)

    # 5. residuals --------------------------------------------------------
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.4 * rows), sharex=True, sharey=True, squeeze=False)
    for ax, m in zip(axes.ravel(), models):
        d = pred[pred["model"] == m].sort_values("date")
        ax.bar(d["date"], d["resid"], width=20 if a.dataset == "fred_monthly" else 1, color=COLORS.get(m))
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{m}: residuals (mean {d['resid'].mean():,.0f})", fontsize=10); ax.grid(alpha=0.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Residual = actual - predicted (positive = model under-forecast)", y=1.0)
    fig.tight_layout(); fig.savefig(fig_dir / "5_residuals.png", dpi=150)

    print("Saved figures to", fig_dir)
    for f in sorted(fig_dir.glob("*.png")):
        print(" -", f.name)
    if a.show:
        matplotlib.use("TkAgg"); plt.show()


if __name__ == "__main__":
    main()

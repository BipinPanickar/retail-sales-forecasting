"""Time-series analysis and forecasting dashboard.   Run:  streamlit run dashboard/app.py

Shows only MEASURED outputs: data in data/processed, results in results/, models in models/.
Nothing on this page is hard-coded.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.analysis import tsa
from src.utils.config import load_config

cfg = load_config()
PROC = ROOT / cfg["paths"]["processed_dir"]
RES = ROOT / cfg["paths"]["results_dir"]
COLORS = {"naive": "#9e9e9e", "seasonal_naive": "#8d6e63", "arima": "#1f77b4", "sarima": "#2ca02c",
          "prophet": "#9467bd", "xgboost": "#ff7f0e", "xgboost_logdiff": "#e6a04a",
          "lstm": "#d62728", "lstm_logdiff": "#8b1a1a"}
ORDER = list(COLORS)

st.set_page_config(page_title="Retail Sales TSA & Forecasting", layout="wide")


# ---------- small helpers --------------------------------------------------
def show(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:                               # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


def table(df):
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


@st.cache_data
def load_series(dataset, series):
    f = PROC / (f"fred_{series}.csv" if dataset == "fred_monthly" else "m5_daily.csv")
    return pd.read_csv(f, index_col=0, parse_dates=True) if f.exists() else None


@st.cache_data
def load_preds(dataset, series):
    f = RES / f"{dataset}_{series}_test_predictions.csv"
    return pd.read_csv(f, parse_dates=["date", "origin"]) if f.exists() else None


@st.cache_data
def load_experiments():
    f = RES / "experiments.csv"
    return pd.read_csv(f) if f.exists() else None


def line_traces(fig, x, y, name, color=None, row=None, col=None, **kw):
    fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color), **kw), row=row, col=col)


def corr_fig(df, col, band, title, m):
    d = df[df["lag"] > 0]
    f = go.Figure(go.Bar(x=d["lag"], y=d[col], marker_color="#1f77b4"))
    for y in (band, -band):
        f.add_hline(y=y, line_dash="dash", line_color="red")
    if m:
        for k in range(m, int(d["lag"].max()) + 1, m):
            f.add_vline(x=k, line_dash="dot", line_color="grey")
    f.update_layout(title=title, xaxis_title="Lag", yaxis_title=col.upper(), height=380, yaxis_range=[-1, 1])
    return f


# ---------- sidebar --------------------------------------------------------
st.sidebar.title("Controls")
dataset = st.sidebar.selectbox("Dataset", ["fred_monthly", "m5_daily"])
if dataset == "fred_monthly":
    series = st.sidebar.selectbox("Series", [s["id"] for s in cfg["fred"]["series"]])
    pred_key = series
else:
    series, pred_key = "total", "total"
dcfg = cfg["datasets"][dataset]
data = load_series(dataset, series)

st.title("Retail Sales: Time-Series Analysis and Forecasting")
st.caption("Research prototype. US Census data is updated MONTHLY (advance estimate, later revised); "
           "this is automated forecasting on regularly updated data, not real-time. M5 is a static benchmark.")

if data is None:
    st.error("Processed data not found. Run:  python -m src.data.ingestion  then  python -m src.data.preprocessing")
    st.stop()

s_full = data["sales"].astype(float)
n_train = int(len(s_full) * cfg["split"]["train_frac"])
window = st.sidebar.radio("Data used for the analysis tabs", ["Train only (recommended)", "Full series"],
                          help="ACF/PACF/stationarity are used to choose model orders, so using the full series "
                               "lets test-period information influence model choice.")
s = s_full.iloc[:n_train] if window.startswith("Train") else s_full
period = st.sidebar.number_input("Seasonal period", 2, 400, int(dcfg["season_length"]),
                                 help="12 for monthly data, 7 for daily data (weekly cycle).")
nlags = st.sidebar.slider("Max lags (ACF/PACF)", 12, 120, 48 if dataset == "fred_monthly" else 60)
st.sidebar.caption(f"{len(s)} observations in view: {s.index.min().date()} to {s.index.max().date()}")

tabs = st.tabs(["Data & trend", "Decomposition & seasonality", "ACF / PACF / lag plot", "Stationarity",
                "Model forecasts", "Residuals", "Metrics & cost", "Future forecast", "LSTM training",
                "Significance & seeds", "XGBoost explainability"])

# ---------- 1. data & trend ------------------------------------------------
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Latest value", f"{s_full.iloc[-1]:,.0f}")
    c[1].metric("Mean", f"{s_full.mean():,.0f}")
    c[2].metric("Observations", len(s_full))
    c[3].metric("Imputed points", int(data["imputed"].sum()) if "imputed" in data else 0)
    wins = st.multiselect("Rolling windows", [3, 6, 7, 12, 14, 28, 30], default=[12] if dataset == "fred_monthly" else [7, 30])
    rs = tsa.rolling_stats(s, wins)
    f = go.Figure()
    f.add_scatter(x=rs.index, y=rs["sales"], name="Sales", line=dict(color="black", width=1))
    for w in wins:
        f.add_scatter(x=rs.index, y=rs[f"mean_{w}"], name=f"Rolling mean {w}")
    f.add_vline(x=s_full.index[n_train], line_dash="dash", line_color="grey", annotation_text="end of train")
    f.update_layout(title="Sales with rolling mean", height=420)
    show(f)
    f2 = go.Figure()
    for w in wins:
        f2.add_scatter(x=rs.index, y=rs[f"std_{w}"], name=f"Rolling std {w}")
    f2.update_layout(title="Rolling standard deviation (does variance grow with the level?)", height=320)
    show(f2)

# ---------- 2. decomposition & seasonality ---------------------------------
with tabs[1]:
    method = st.radio("Method", ["STL", "Classical additive", "Classical multiplicative"], horizontal=True)
    try:
        if method == "STL":
            dec = tsa.stl_decompose(s, int(period), robust=True)
        else:
            dec = tsa.classical_decompose(s, int(period), "additive" if "additive" in method else "multiplicative")
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])
        for i, comp in enumerate(["observed", "trend", "seasonal", "resid"], start=1):
            line_traces(fig, dec.index, dec[comp], comp, row=i, col=1)
        fig.update_layout(height=760, showlegend=False)
        show(fig)
        if method != "Classical multiplicative":
            st_ = tsa.component_strength(dec)
            c = st.columns(2)
            c[0].metric("Trend strength (0-1)", f"{st_['trend_strength']:.2f}")
            c[1].metric("Seasonal strength (0-1)", f"{st_['seasonal_strength']:.2f}")
    except Exception as e:
        st.error(f"Decomposition failed: {e}")
    st.subheader("Seasonality patterns")
    if dataset == "fred_monthly":
        d = s.to_frame("sales"); d["year"], d["month"] = d.index.year, d.index.month
        piv = d.pivot_table(index="month", columns="year", values="sales")
        f = px.line(piv, title="Seasonal plot: one line per year (same shape each year = stable seasonality)")
        f.update_layout(height=420, xaxis_title="Month"); show(f)
        show(px.box(d, x="month", y="sales", title="Distribution by calendar month"))
    else:
        d = s.to_frame("sales"); d["weekday"] = d.index.day_name()
        show(px.box(d, x="weekday", y="sales", title="Distribution by weekday"))
        d["month"] = d.index.month
        show(px.box(d, x="month", y="sales", title="Distribution by month"))

# ---------- 3. ACF / PACF / lag plot ---------------------------------------
with tabs[2]:
    base = st.radio("Series", ["Original", "First difference (d=1)", "Seasonal difference (D=1)", "Both (d=1, D=1)"],
                    horizontal=True)
    d_, D_ = {"Original": (0, 0), "First difference (d=1)": (1, 0),
              "Seasonal difference (D=1)": (0, 1), "Both (d=1, D=1)": (1, 1)}[base]
    x = tsa.diff_series(s, d_, D_, int(period))
    try:
        ap, band = tsa.acf_pacf(x, nlags)
        c1, c2 = st.columns(2)
        with c1:
            show(corr_fig(ap, "acf", band, f"ACF - {base}", int(period)))
        with c2:
            show(corr_fig(ap, "pacf", band, f"PACF - {base}", int(period)))
        st.caption("Red dashed = approximate 95% white-noise band (1.96/sqrt(n)); grey dotted = multiples of the "
                   "seasonal period. Lag 0 is omitted (always 1).")
    except Exception as e:
        st.error(f"ACF/PACF failed: {e}")
    with st.expander("How to read ACF/PACF for ARIMA/SARIMA"):
        st.markdown(
            "- Slowly decaying ACF -> trend / non-stationarity -> consider differencing.\n"
            "- Spikes at lag m, 2m, ... -> seasonality of period m -> seasonal terms (P, Q) or seasonal differencing.\n"
            "- PACF cuts off after lag p and ACF decays -> AR(p) pattern; ACF cuts off after lag q and PACF decays -> MA(q).\n"
            "- These are guides, not rules: the final orders are chosen by AIC and validation error.")
    st.subheader("Autocorrelation (lag) plot")
    k = st.slider("Lag k", 1, int(max(period * 2, 24)), int(period))
    lagdf = pd.DataFrame({"y_t": x.values[k:], f"y_t-{k}": x.values[:-k]})
    r = np.corrcoef(lagdf.iloc[:, 0], lagdf.iloc[:, 1])[0, 1]
    f = px.scatter(lagdf, x=f"y_t-{k}", y="y_t", title=f"y(t) vs y(t-{k}); correlation = {r:.3f}")
    f.update_layout(height=420); show(f)

# ---------- 4. stationarity ------------------------------------------------
with tabs[3]:
    st.markdown("**ADF** H0: unit root (non-stationary), small p -> stationary.  "
                "**KPSS** H0: stationary, small p -> NON-stationary. Using both guards against either test's weakness.")
    rows = []
    for label, (dd, DD) in {"Original": (0, 0), "d=1": (1, 0), "D=1 (seasonal)": (0, 1), "d=1 and D=1": (1, 1)}.items():
        try:
            xx = tsa.diff_series(s, dd, DD, int(period))
            a, kp = tsa.adf_test(xx), tsa.kpss_test(xx)
            rows.append({"Series": label, "n": len(xx), "ADF stat": round(a["stat"], 3), "ADF p": round(a["p"], 4),
                         "KPSS stat": round(kp["stat"], 3), "KPSS p (bounded 0.01-0.10)": round(kp["p"], 3),
                         "Verdict": tsa.stationarity_verdict(a["p"], kp["p"])})
        except Exception as e:
            rows.append({"Series": label, "Verdict": f"failed: {e}"})
    table(pd.DataFrame(rows))
    st.caption("A structural break (e.g. 2020) can distort both tests. Report conflicts honestly instead of picking one.")

# ---------- 5. model forecasts ---------------------------------------------
preds = load_preds(dataset, pred_key)
with tabs[4]:
    if preds is None:
        st.warning("No predictions yet. Run: python -m src.run_experiments --dataset " + dataset +
                   (f" --series {series}" if dataset == "fred_monthly" else ""))
    else:
        avail = [m for m in ORDER if m in set(preds["model"])]
        picks = st.multiselect("Models", avail, default=[m for m in ["sarima", "xgboost_logdiff", "lstm_logdiff"] if m in avail] or avail[:3])
        ctx_years = st.slider("History shown before test (years)", 1, 15, 6)
        test_start = preds["date"].min()
        ctx = s_full[s_full.index >= test_start - pd.DateOffset(years=ctx_years)]
        f = go.Figure()
        f.add_scatter(x=ctx.index, y=ctx.values, name="Actual", line=dict(color="black", width=2))
        for m in picks:
            for i, (_, b) in enumerate(preds[preds["model"] == m].groupby("origin")):
                b = b.sort_values("date")
                f.add_scatter(x=b["date"], y=b["pred"], name=m, legendgroup=m, showlegend=(i == 0),
                              line=dict(color=COLORS.get(m)))
        f.update_layout(title="Test-period forecasts (each segment = one block forecast from real history)", height=520)
        show(f)
        st.subheader("One panel per model")
        sel = [m for m in avail] if st.checkbox("Show all models", value=True) else picks
        if sel:
            cols = 2
            rows_n = int(np.ceil(len(sel) / cols))
            fig = make_subplots(rows=rows_n, cols=cols, subplot_titles=sel, shared_xaxes=True, shared_yaxes=True)
            act = preds[preds["model"] == sel[0]].sort_values("date")
            for i, m in enumerate(sel):
                r_, c_ = i // cols + 1, i % cols + 1
                line_traces(fig, act["date"], act["actual"], "Actual", "black", r_, c_, showlegend=(i == 0))
                for _, b in preds[preds["model"] == m].groupby("origin"):
                    b = b.sort_values("date")
                    line_traces(fig, b["date"], b["pred"], m, COLORS.get(m), r_, c_, showlegend=False)
            fig.update_layout(height=320 * rows_n)
            show(fig)

# ---------- 6. residuals ----------------------------------------------------
with tabs[5]:
    if preds is None:
        st.info("Run the experiments first.")
    else:
        m = st.selectbox("Model", [x for x in ORDER if x in set(preds["model"])], key="resid_model",
                         index=min(3, len(set(preds["model"])) - 1))
        d = preds[preds["model"] == m].sort_values("date").copy()
        d["resid"] = d["actual"] - d["pred"]
        c = st.columns(3)
        c[0].metric("Mean residual (bias)", f"{d['resid'].mean():,.0f}")
        c[1].metric("Std of residual", f"{d['resid'].std():,.0f}")
        try:
            lb = tsa.ljung_box_p(d["resid"], int(min(period, 12)))
            c[2].metric(f"Ljung-Box p (lag {int(min(period, 12))})", f"{lb:.3f}")
        except Exception:
            pass
        show(px.bar(d, x="date", y="resid", title=f"{m}: residual = actual - predicted (positive = under-forecast)"))
        c1, c2 = st.columns(2)
        with c1:
            show(px.histogram(d, x="resid", nbins=25, title="Residual distribution"))
        with c2:
            try:
                ap, band = tsa.acf_pacf(d["resid"], min(24, len(d) // 2 - 2))
                show(corr_fig(ap, "acf", band, "ACF of residuals", None))
            except Exception as e:
                st.info(f"ACF of residuals unavailable: {e}")
        st.caption("Residuals come from multi-step block forecasts, so some autocorrelation is expected even for a "
                   "good model; a persistent one-sided bias is the more serious warning sign.")

# ---------- 7. metrics & cost ------------------------------------------------
with tabs[6]:
    exp = load_experiments()
    if exp is None:
        st.info("No experiments yet.")
    else:
        e = exp[(exp["dataset"] == dataset) & (exp["series"] == series)]
        if e.empty:
            st.info("No experiments for this dataset/series yet.")
        else:
            runs = e.groupby("experiment_id").agg(ts=("timestamp", "max"), note=("notes", "first")).sort_values("ts", ascending=False)
            eid = st.selectbox("Experiment", runs.index,
                               format_func=lambda i: f"{i} - {'quick' if runs.loc[i, 'note'] == 'quick' else 'full'} - {runs.loc[i, 'ts']}")
            run = e[e["experiment_id"] == eid]
            cols_ = ["model", "split", "MAE", "RMSE", "sMAPE", "WAPE", "training_time_s"]
            table(run[cols_].sort_values(["split", "MAE"]).reset_index(drop=True))
            test = run[run["split"] == "test"]
            c1, c2 = st.columns(2)
            with c1:
                show(px.bar(test.sort_values("MAE"), x="model", y="MAE", color="model", color_discrete_map=COLORS, title="Test MAE"))
            with c2:
                show(px.scatter(test, x="training_time_s", y="MAE", color="model", color_discrete_map=COLORS, text="model",
                                title="Accuracy vs training cost (lower-left is better)"))
            if preds is not None:
                p = preds.assign(abs_err=(preds["actual"] - preds["pred"]).abs())
                g = p.groupby(["model", "step"], as_index=False)["abs_err"].mean()
                show(px.line(g, x="step", y="abs_err", color="model", color_discrete_map=COLORS, markers=True,
                             title="Mean absolute error by forecast step (horizon effect)"))
            st.caption("Single split and single seed: treat small differences with caution; significance testing is a later step.")

# ---------- 8. future forecast -----------------------------------------------
with tabs[7]:
    if dataset != "fred_monthly":
        st.info("M5 is a static benchmark; future forecasts are produced for the live FRED series only.")
    else:
        model_name = st.selectbox("Model", ORDER, index=ORDER.index("sarima"), key="future_model")
        horizon = st.selectbox("Horizon (months)", [3, 6, 12, 24], index=2)
        st.caption("Saved models were fitted on the TRAIN period and are conditioned on the latest data at forecast time "
                   "(no re-estimation). Prophet extrapolates its fitted trend instead.")
        if st.button("Generate forecast"):
            try:
                from src.forecasting.forecast import generate_forecast
                fc = generate_forecast(series, model_name, horizon)
                f = go.Figure()
                f.add_scatter(x=s_full.index[-60:], y=s_full.values[-60:], name="History", line=dict(color="black"))
                f.add_scatter(x=fc["forecast_date"], y=fc["predicted_sales"], name=model_name, line=dict(color=COLORS.get(model_name)))
                f.update_layout(title=f"{model_name}: next {horizon} months", height=450)
                show(f)
                table(fc)
            except Exception as e:
                st.error(f"Could not forecast (is the model trained and saved?): {e}")

# ---------- 9. LSTM training --------------------------------------------------
with tabs[8]:
    import json
    found = False
    for variant in ["lstm", "lstm_logdiff"]:
        mdir = ROOT / cfg["paths"]["models_dir"] / variant / f"{dataset}_{pred_key}_{variant}"
        meta_f = (mdir / "inner" / "meta.json") if variant.endswith("logdiff") else (mdir / "meta.json")
        if not meta_f.exists():
            continue
        found = True
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        st.subheader(variant)
        c = st.columns(3)
        c[0].metric("Lookback window", meta.get("lookback"))
        c[1].metric("Epochs run (early stopping)", meta.get("epochs_run"))
        c[2].metric("Inputs", 1 + len(meta.get("exog_cols", [])))
        h = meta.get("history")
        if h:
            f = go.Figure()
            f.add_scatter(y=h["loss"], name="train loss")
            if "val_loss" in h:
                f.add_scatter(y=h["val_loss"], name="validation loss")
            f.update_layout(title=f"{variant}: training curves (MSE on the scaled target)", xaxis_title="Epoch", height=380)
            show(f)
        else:
            st.info("Training history not stored for this model. Re-run the experiments to record it.")
    if not found:
        st.info("No saved LSTM yet. Run the experiments including the lstm models.")
    else:
        st.caption("LSTM has no built-in feature importance comparable to XGBoost/SHAP. Loss curves are on each model's own "
                   "scaled target, so compare models by MAE/WAPE, not by loss.")

# ---------- 10. significance & seeds -----------------------------------------------
with tabs[9]:
    tag = f"{dataset}_{pred_key}"
    dmf, sdf = RES / f"{tag}_dm_tests.csv", RES / f"{tag}_seeds.csv"
    st.subheader("Diebold-Mariano tests vs a reference model")
    if dmf.exists():
        dm = pd.read_csv(dmf)
        table(dm.round(4))
        st.caption("Negative mean_loss_diff = model has lower loss than the reference. p_holm is Holm-adjusted across models. "
                   "With ~60 test points power is low: a non-significant result is not proof of equality.")
    else:
        st.info(f"Not run yet:  python -m src.evaluation.significance --dataset {dataset}"
                + (f" --series {series}" if dataset == "fred_monthly" else ""))
    st.subheader("Stability across random seeds")
    if sdf.exists():
        sd = pd.read_csv(sdf)
        table(sd.groupby("model")[["MAE", "WAPE"]].agg(["mean", "std", "min", "max"]).round(2))
        show(px.box(sd, x="model", y="MAE", points="all", color="model", color_discrete_map=COLORS, title="Test MAE across seeds"))
    else:
        st.info(f"Not run yet:  python -m src.run_seeds --dataset {dataset}"
                + (f" --series {series}" if dataset == "fred_monthly" else "") + " --seeds 5")

# ---------- 11. XGBoost explainability ------------------------------------------------
with tabs[10]:
    st.caption("Native XGBoost TreeSHAP on one-step-ahead feature rows built from actual lags (explains the fitted model, "
               "not the recursive forecast path; descriptive, not causal).")
    variant = st.radio("Model", ["xgboost", "xgboost_logdiff"], horizontal=True, key="shap_model")
    mdir = ROOT / cfg["paths"]["models_dir"] / variant / f"{dataset}_{pred_key}_{variant}"
    mdir = mdir / "inner" if variant.endswith("logdiff") else mdir
    if not (mdir / "meta.json").exists():
        st.info("Model not found. Run the experiments first.")
    else:
        try:
            from src.evaluation.explain import xgb_shap, mean_abs_shap
            from src.models.logdiff import to_logdiff
            from src.models.xgboost_model import XGBoostForecaster
            mdl = XGBoostForecaster.load(mdir)
            frame = data.drop(columns=[c for c in ["imputed"] if c in data.columns])
            frame = to_logdiff(frame) if variant.endswith("logdiff") else frame
            test_start = s_full.index[n_train + int(len(s_full) * cfg["split"]["val_frac"])]
            sh, base = xgb_shap(mdl, frame, 0, len(frame))
            sh = sh[sh.index >= test_start]
            imp = mean_abs_shap(sh)
            show(px.bar(imp.iloc[::-1], orientation="h", title="Mean |SHAP| over the test period (target units of this model)"))
            top = st.multiselect("Features over time", list(imp.index), default=list(imp.index[:4]))
            if top:
                show(px.line(sh[top], title="SHAP value per test row"))
            st.caption("For xgboost_logdiff the target is log growth, so SHAP units are log-points, not sales units.")
        except Exception as e:
            st.error(f"Explainability failed: {e}")

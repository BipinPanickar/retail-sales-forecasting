"""XGBoost explainability using XGBoost's NATIVE TreeSHAP (pred_contribs=True): no `shap`
package needed. SHAP values are computed on one-step-ahead feature rows built from ACTUAL lags
(not recursive forecasts), so they explain the fitted model, not the multi-step forecast path.
Mean |SHAP| ranks features; it is a description of the model, not a causal statement.
"""
import numpy as np
import pandas as pd

from src.features.feature_engineering import make_features


def xgb_shap(model, df: pd.DataFrame, start: int, end: int):
    """model: fitted XGBoostForecaster; df: frame it was trained on (same transform);
    rows [start, end) are explained. Returns (shap_df, base_value)."""
    import xgboost as xgb
    X = make_features(df, model.lags, model.windows, model.exog_cols, model.freq)[model.columns]
    X = X.iloc[start:end].dropna()
    contrib = model.model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    shap_df = pd.DataFrame(contrib[:, :-1], index=X.index, columns=model.columns)
    return shap_df, float(contrib[0, -1])


def mean_abs_shap(shap_df: pd.DataFrame) -> pd.Series:
    return shap_df.abs().mean().sort_values(ascending=False)

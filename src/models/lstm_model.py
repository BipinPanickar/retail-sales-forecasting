"""LSTM forecaster (Keras). Input = sliding window of `lookback` past rows.

Leakage controls:
  * MinMaxScaler is fitted on TRAIN sales only; val/test/hist are only transformed.
  * Validation windows use targets in the validation period; inputs are strictly earlier rows.
  * Exogenous inputs must be known in advance (calendar flags). Binary flags are left unscaled.
Known limitation: on a strongly trending series, a bounded-scaled LSTM extrapolates poorly.
Interpretability: LSTM has no built-in feature importance comparable to XGBoost/SHAP.
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation.backtest import rolling_origin, get_exog_cols
from src.evaluation.metrics import mae
from src.models.base import BaseForecaster, write_json, read_json
from src.utils.logger import get_logger

log = get_logger("lstm")


def make_sequences(arr: np.ndarray, lookback: int, start: int, end: int):
    """Targets are arr[t, 0] for t in [start, end); inputs are arr[t-lookback:t]."""
    X = np.stack([arr[t - lookback:t] for t in range(max(start, lookback), end)])
    y = np.array([arr[t, 0] for t in range(max(start, lookback), end)])
    return X, y


class LSTMForecaster(BaseForecaster):
    name = "lstm"

    def __init__(self, lookback, exog_cols=(), cfg=None, seed=42):
        self.lookback, self.exog_cols, self.cfg, self.seed = lookback, list(exog_cols), cfg or {}, seed
        self.scaler, self.model, self.epochs_run, self.history_ = None, None, None, None

    def _array(self, df):
        s = self.scaler.transform(df[["sales"]]).ravel()
        return np.column_stack([s] + [df[c].values.astype(float) for c in self.exog_cols])

    def _build(self, n_features):
        import tensorflow as tf
        from tensorflow.keras import layers, models, optimizers
        c = self.cfg
        m = models.Sequential([
            layers.Input(shape=(self.lookback, n_features)),
            layers.LSTM(c["units_1"], return_sequences=True),
            layers.Dropout(c["dropout"]),
            layers.LSTM(c["units_2"]),
            layers.Dense(c["dense"], activation="relu"),
            layers.Dense(1)])
        m.compile(optimizer=optimizers.Adam(c["learning_rate"]), loss="mse")
        return m

    def fit(self, train, val=None, ckpt_path: Path = None):
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras import callbacks
        tf.keras.utils.set_random_seed(self.seed)
        self.scaler = MinMaxScaler().fit(train[["sales"]])        # TRAIN ONLY
        full = pd.concat([train, val])
        arr = self._array(full)
        n_tr = len(train)
        Xtr, ytr = make_sequences(arr, self.lookback, self.lookback, n_tr)
        Xva, yva = make_sequences(arr, self.lookback, n_tr, len(full))
        self.model = self._build(arr.shape[1])
        cbs = [callbacks.EarlyStopping(monitor="val_loss", patience=self.cfg["patience"], restore_best_weights=True)]
        if ckpt_path is not None:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            cbs.append(callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True))
        hist = self.model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=self.cfg["epochs"],
                              batch_size=self.cfg["batch_size"], callbacks=cbs, verbose=0, shuffle=False)
        self.epochs_run = len(hist.history["loss"])
        self.history_ = {k: [float(x) for x in v] for k, v in hist.history.items()}
        return self

    def predict(self, hist, future_exog):
        arr = self._array(hist)
        window = arr[-self.lookback:].copy()
        out = []
        for i in range(len(future_exog)):
            yhat = float(self.model(window[None, :, :], training=False).numpy()[0, 0])
            out.append(yhat)
            ex = future_exog[self.exog_cols].iloc[i].values.astype(float) if self.exog_cols else np.array([])
            window = np.vstack([window[1:], np.concatenate([[yhat], ex])])   # recursive
        return self.scaler.inverse_transform(np.array(out).reshape(-1, 1)).ravel()

    def describe(self):
        return {**self.cfg, "lookback": self.lookback, "epochs_run": self.epochs_run}

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory / "model.keras")
        joblib.dump(self.scaler, directory / "scaler.joblib")
        write_json(directory / "meta.json", {"model": self.name, "lookback": self.lookback,
                                             "exog_cols": self.exog_cols, "cfg": self.cfg, "seed": self.seed,
                                             "epochs_run": self.epochs_run, "history": self.history_})

    @classmethod
    def load(cls, directory: Path):
        from tensorflow.keras.models import load_model
        m = read_json(directory / "meta.json")
        obj = cls(m["lookback"], m["exog_cols"], m["cfg"], m["seed"])
        obj.model = load_model(directory / "model.keras")
        obj.scaler = joblib.load(directory / "scaler.joblib")
        obj.epochs_run = m.get("epochs_run")
        obj.history_ = m.get("history")
        return obj

    @staticmethod
    def select(train, val, horizon, dcfg, lcfg, seed=42, quick=False, ckpt_dir: Path = None, logdiff=False):
        exog = get_exog_cols(train)
        if logdiff:
            from src.models.logdiff import split_logdiff, LSTMLogDiff
            train_t, val_t = split_logdiff(train, val)
        else:
            train_t, val_t = train, val
        lookbacks = dcfg["lstm_lookbacks"][:1] if quick else dcfg["lstm_lookbacks"]
        cfg = {**lcfg, "epochs": min(lcfg["epochs"], 15) if quick else lcfg["epochs"]}
        full = pd.concat([train, val])
        best, best_mae = None, np.inf
        for L in lookbacks:
            t0 = time.time()
            m = LSTMForecaster(L, exog, cfg, seed).fit(
                train_t, val_t, ckpt_path=(ckpt_dir / f"ckpt_L{L}{'_ld' if logdiff else ''}.keras") if ckpt_dir else None)
            m = LSTMLogDiff(m) if logdiff else m
            r = rolling_origin(m, full, len(train), len(full), horizon)
            score = mae(r["actual"], r["pred"])
            inner = m.inner if logdiff else m
            log.info("LSTM%s lookback=%d epochs=%d val MAE=%.3f (%.0fs)", "(logdiff)" if logdiff else "", L, inner.epochs_run, score, time.time() - t0)
            if score < best_mae:
                best, best_mae = m, score
        return best

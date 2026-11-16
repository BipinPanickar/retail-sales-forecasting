"""Raw -> processed time series for each dataset.

Imputation policy (explained, not blind zero-filling):
  * Monthly FRED: missing months are short gaps -> linear interpolation, flagged in 'imputed'.
  * Sales are NEVER filled with 0 unless the source says there were zero sales.
"""
import pandas as pd

from src.data.validation import check_missing_dates
from src.utils.config import load_config, resolve
from src.utils.logger import get_logger
from src.data.ingestion import get_connection

log = get_logger("preprocessing")


def load_fred_series(series_id: str, as_of: str | None = None) -> pd.Series:
    """Latest known value per month. If as_of is given, only data ingested up to that
    timestamp is used (point-in-time view; avoids revision look-ahead)."""
    cfg = load_config()
    conn = get_connection(cfg)
    q = "SELECT date, sales, ingested_at, id FROM raw_sales WHERE series_id=?"
    df = pd.read_sql_query(q, conn, params=(series_id,))
    conn.close()
    if df.empty:
        raise ValueError(f"No data for {series_id}. Run: python -m src.data.ingestion")
    if as_of:
        df = df[df["ingested_at"] <= as_of]
    df = df.sort_values("id").drop_duplicates("date", keep="last")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["sales"].sort_index().rename("sales")


def clean_series(s: pd.Series, freq: str, name: str = "sales") -> pd.DataFrame:
    """Reindex to a complete calendar, interpolate short gaps, flag imputed points."""
    s = s[~s.index.duplicated(keep="last")].sort_index()
    missing = check_missing_dates(s, freq)
    full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq=freq))
    imputed = full.isna()
    n_gap = int(imputed.sum())
    if n_gap:
        log.info("%s: %d missing dates found -> linear interpolation (flagged).", name, n_gap)
        full = full.interpolate(method="linear", limit=3)
    if full.isna().any():
        raise ValueError("Gaps longer than 3 periods remain; investigate manually.")
    if (full < 0).any():
        raise ValueError("Negative sales detected; investigate before modelling.")
    out = pd.DataFrame({"sales": full, "imputed": imputed.astype(int)})
    out.index.name = "date"
    return out


def build_fred_dataset(series_id: str) -> pd.DataFrame:
    cfg = load_config()
    df = clean_series(load_fred_series(series_id), cfg["datasets"]["fred_monthly"]["freq"], series_id)
    df.to_csv(resolve(f"{cfg['paths']['processed_dir']}/fred_{series_id}.csv"))
    return df


if __name__ == "__main__":
    cfg = load_config()
    for s in cfg["fred"]["series"]:
        try:
            d = build_fred_dataset(s["id"])
            print(s["id"], d.shape, d.index.min().date(), "->", d.index.max().date())
        except ValueError as e:
            print(f"Skipped {s['id']}: {e}")

"""Incremental, idempotent ingestion of FRED retail-sales data into SQLite.

Design:
  * First run  -> full history from config.fred.history_start.
  * Later runs -> only recent months (lookback window) to catch new data AND revisions.
  * A row is inserted only if (series_id, date) is new OR its value changed.
    Changed values are stored as a NEW version (old one kept) -> revision history.
  * Source frequency is MONTHLY: most scheduled runs will find nothing new. That is normal.
"""
import os
import sqlite3
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

from src.data.validation import validate_fred_observations, ValidationError
from src.utils.config import load_config, resolve, PROJECT_ROOT
from src.utils.logger import get_logger

log = get_logger("ingestion")

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_sales (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id    TEXT NOT NULL,
    date         TEXT NOT NULL,
    sales        REAL,
    source       TEXT NOT NULL,
    ingested_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_raw_series_date ON raw_sales(series_id, date);

CREATE TABLE IF NOT EXISTS forecasts (
    forecast_date  TEXT NOT NULL,
    series_id      TEXT NOT NULL,
    model_name     TEXT NOT NULL,
    predicted_sales REAL NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingestion_log (
    run_at TEXT, series_id TEXT, fetched INTEGER, inserted INTEGER, revised INTEGER, status TEXT
);
"""


def get_connection(cfg: dict) -> sqlite3.Connection:
    conn = sqlite3.connect(resolve(cfg["paths"]["db"]))
    conn.executescript(SCHEMA)
    return conn


def _latest_values(conn, series_id: str) -> pd.DataFrame:
    """Latest stored version of each date for one series."""
    q = """SELECT date, sales FROM raw_sales WHERE series_id=? AND id IN
           (SELECT MAX(id) FROM raw_sales WHERE series_id=? GROUP BY date)"""
    df = pd.read_sql_query(q, conn, params=(series_id, series_id))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["sales"]


def fetch_fred(series_id: str, start: str, cfg: dict, api_key: str) -> pd.DataFrame:
    """Call the FRED API with retries. Raises on persistent failure."""
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json",
              "observation_start": start}
    fc = cfg["fred"]
    last_err = None
    for attempt in range(1, fc["max_retries"] + 1):
        try:
            r = requests.get(fc["base_url"], params=params, timeout=fc["timeout_seconds"])
            if r.status_code == 429:
                raise requests.HTTPError("Rate limited (429)")
            r.raise_for_status()
            return validate_fred_observations(r.json().get("observations", []))
        except (requests.RequestException, ValidationError, ValueError) as e:
            last_err = e
            log.warning("Attempt %d/%d failed for %s: %s", attempt, fc["max_retries"], series_id, e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"FRED fetch failed for {series_id}: {last_err}")


def store_observations(conn, series_id: str, df: pd.DataFrame, source: str = "FRED"):
    """Insert new rows and changed (revised) rows. Return (inserted, revised)."""
    existing = _latest_values(conn, series_id)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    inserted = revised = 0
    rows = []
    for d, v in zip(df["date"], df["value"]):
        if d not in existing.index:
            inserted += 1
        else:
            old = existing.loc[d]
            same = (pd.isna(old) and pd.isna(v)) or (not pd.isna(old) and not pd.isna(v) and abs(old - v) < 1e-9)
            if same:
                continue
            revised += 1
        rows.append((series_id, d.strftime("%Y-%m-%d"), None if pd.isna(v) else float(v), source, now))
    conn.executemany("INSERT INTO raw_sales(series_id,date,sales,source,ingested_at) VALUES (?,?,?,?,?)", rows)
    conn.commit()
    return inserted, revised


def run_ingestion() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    api_key = os.getenv("FRED_API_KEY")
    if not api_key or "your_fred" in api_key:
        raise SystemExit("FRED_API_KEY missing. Copy .env.example to .env and add your key.")
    cfg = load_config()
    conn = get_connection(cfg)
    for s in cfg["fred"]["series"]:
        sid = s["id"]
        existing = _latest_values(conn, sid)
        if existing.empty:
            start = cfg["fred"]["history_start"]
            log.info("%s: first run, downloading full history from %s", sid, start)
        else:
            start = (existing.index.max() - pd.DateOffset(months=cfg["fred"]["revision_lookback_months"])).strftime("%Y-%m-%d")
            log.info("%s: incremental run from %s", sid, start)
        try:
            df = fetch_fred(sid, start, cfg, api_key)
            ins, rev = store_observations(conn, sid, df)
            status = "ok"
            log.info("%s: fetched=%d inserted=%d revised=%d", sid, len(df), ins, rev)
        except Exception as e:  # keep other series running
            log.error("%s failed: %s", sid, e)
            df, ins, rev, status = pd.DataFrame(), 0, 0, f"error: {e}"
        conn.execute("INSERT INTO ingestion_log VALUES (?,?,?,?,?,?)",
                     (datetime.now(timezone.utc).isoformat(timespec="seconds"), sid, len(df), ins, rev, status))
        conn.commit()
    conn.close()


if __name__ == "__main__":
    run_ingestion()

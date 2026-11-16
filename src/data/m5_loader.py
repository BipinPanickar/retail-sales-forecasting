"""Load the static M5 (Walmart) data and aggregate to a daily series.

IMPORTANT: M5 is a fixed Kaggle file. It is NOT live. Target = daily UNIT sales
(not revenue) unless prices are joined. Calendar variables (weekday, events, SNAP)
are known in advance, so they are legitimate exogenous features.
Download from Kaggle (M5 Forecasting - Accuracy) into data/external/m5/ yourself.
"""
import pandas as pd

from src.utils.config import load_config, resolve, PROJECT_ROOT
from src.utils.logger import get_logger

log = get_logger("m5")


def load_m5_daily() -> pd.DataFrame:
    cfg = load_config()
    d = PROJECT_ROOT / cfg["paths"]["m5_dir"]
    sales_path = d / cfg["m5"]["sales_file"]
    if not sales_path.exists():
        alt = d / "sales_train_validation.csv"
        if alt.exists():
            sales_path = alt
        else:
            raise FileNotFoundError(f"Place M5 CSV files in {d} (see README).")
    sales = pd.read_csv(sales_path)
    cal = pd.read_csv(d / cfg["m5"]["calendar_file"])
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    level = cfg["m5"]["level"]
    if level == "total":
        daily = sales[day_cols].sum(axis=0)
    elif level in ("state", "store"):
        raise NotImplementedError("Only level: total is implemented; extend here if needed.")
    else:
        raise ValueError(level)
    df = daily.rename("sales").rename_axis("d").reset_index()
    df = df.merge(cal[["d", "date", "wday", "month", "year", "event_name_1",
                       "snap_CA", "snap_TX", "snap_WI"]], on="d", how="left")
    df["date"] = pd.to_datetime(df["date"])
    df["has_event"] = df["event_name_1"].notna().astype(int)
    df["snap_any"] = df[["snap_CA", "snap_TX", "snap_WI"]].max(axis=1)
    df = df.set_index("date").sort_index()[["sales", "has_event", "snap_any"]]
    df.to_csv(resolve(f"{cfg['paths']['processed_dir']}/m5_daily.csv"))
    log.info("M5 daily series: %d days, %s -> %s", len(df), df.index.min().date(), df.index.max().date())
    return df


if __name__ == "__main__":
    print(load_m5_daily().tail())

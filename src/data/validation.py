"""Validation of raw API/CSV data before it touches the database."""
import pandas as pd


class ValidationError(Exception):
    pass


def validate_fred_observations(obs: list) -> pd.DataFrame:
    """Validate the 'observations' list from a FRED response.

    FRED encodes missing values as '.', which we convert to NaN (not zero).
    """
    if not isinstance(obs, list) or len(obs) == 0:
        raise ValidationError("FRED response has no observations.")
    df = pd.DataFrame(obs)
    for col in ("date", "value"):
        if col not in df.columns:
            raise ValidationError(f"Missing column '{col}' in FRED response.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValidationError("Unparseable dates in FRED response.")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # "." (FRED missing) -> NaN
    if (df["value"].dropna() < 0).any():
        raise ValidationError("Negative retail sales value found - investigate before storing.")
    return df[["date", "value"]]


def check_missing_dates(s: pd.Series, freq: str) -> pd.DatetimeIndex:
    """Return the dates missing from a series given its expected frequency."""
    full = pd.date_range(s.index.min(), s.index.max(), freq=freq)
    return full.difference(s.index)

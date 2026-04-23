from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

TICKER = "GC=F"


def download_gold_data() -> pd.DataFrame:
    """
    Download daily gold futures data from Yahoo Finance
    and normalize it to the raw contract expected by the pipeline:
    Date, Open, High, Low, Close
    """
    df = yf.download(TICKER, period="max", interval="1d", auto_adjust=False)

    if df.empty:
        raise ValueError("Downloaded data is empty.")

    # yfinance may return MultiIndex columns like:
    # ('Close', 'GC=F'), ('Open', 'GC=F'), etc.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    required_cols = ["Date", "Open", "High", "Low", "Close"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Downloaded data is missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[required_cols].copy()

    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"]).reset_index(
        drop=True
    )

    return df


def get_expected_market_date() -> str:
    """
    Heuristic for expected latest daily market date.

    Rules:
    - Saturday  -> expected latest date = Friday
    - Sunday    -> expected latest date = Friday
    - Weekday before 22:00 UTC -> expected latest date = previous business day
    - Weekday after 22:00 UTC  -> expected latest date = today

    This avoids unnecessary downloads while still allowing refresh
    once a new daily bar is likely available.
    """
    now = datetime.utcnow()
    current_date = now.date()

    # Weekend -> previous Friday
    if current_date.weekday() == 5:  # Saturday
        current_date = current_date - timedelta(days=1)
    elif current_date.weekday() == 6:  # Sunday
        current_date = current_date - timedelta(days=2)

    # Before daily data is likely settled -> use previous business day
    if now.hour < 22:
        current_date = current_date - timedelta(days=1)
        while current_date.weekday() >= 5:
            current_date = current_date - timedelta(days=1)

    return current_date.strftime("%Y-%m-%d")


def collect_gold_data(cache_dir: str = "data/raw") -> pd.DataFrame:
    """
    Freshness-aware ingestion based on the latest market date,
    not file modification time.

    Logic:
    - if cache exists and its latest date is current enough -> use cache
    - otherwise try downloading fresh data
    - if download fails and cache exists -> fallback to cache
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    file_path = Path(cache_dir) / "xauusd_latest.csv"

    expected_latest_date = get_expected_market_date()

    local_df = None
    if file_path.exists():
        local_df = pd.read_csv(file_path)

        if not local_df.empty and "Date" in local_df.columns:
            local_last_date = str(local_df["Date"].iloc[-1])

            if local_last_date >= expected_latest_date:
                print(
                    f"Using cached gold data. "
                    f"Cache date {local_last_date} is current enough "
                    f"(expected >= {expected_latest_date})."
                )
                return local_df

    try:
        remote_df = download_gold_data()
    except Exception as e:
        if local_df is not None and not local_df.empty:
            print(f"Warning: live download failed, using cached data. Reason: {e}")
            return local_df
        raise RuntimeError(
            f"Failed to download gold data and no cache is available: {e}"
        ) from e

    if remote_df.empty:
        if local_df is not None and not local_df.empty:
            print("Warning: downloaded dataframe is empty, using cached data.")
            return local_df
        raise RuntimeError("Downloaded dataframe is empty and no cache is available.")

    remote_last_date = str(remote_df["Date"].iloc[-1])

    if local_df is not None and not local_df.empty and "Date" in local_df.columns:
        local_last_date = str(local_df["Date"].iloc[-1])

        if local_last_date == remote_last_date:
            print("Remote data unchanged. Using existing file.")
            return local_df

    remote_df.to_csv(file_path, index=False)
    print(
        f"Downloaded and updated gold data. " f"Latest remote date: {remote_last_date}"
    )

    return remote_df


if __name__ == "__main__":
    df = collect_gold_data()
    print(f"[ingest] available rows: {len(df)}")
    print(f"[ingest] latest date: {df['Date'].iloc[-1]}")

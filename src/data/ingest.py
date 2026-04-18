from pathlib import Path
import time

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

    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"]).reset_index(drop=True)

    return df


def collect_gold_data(cache_dir: str = "data/raw") -> pd.DataFrame:
    """
    Freshness-aware ingestion:
    - if local cache is younger than 24h -> use cache
    - otherwise try live download
    - if remote latest date == local latest date -> keep local file
    - if live download fails and cache exists -> fallback to cache
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    file_path = Path(cache_dir) / "xauusd_latest.csv"

    if file_path.exists():
        file_age = time.time() - file_path.stat().st_mtime
        if file_age < 24 * 3600:
            print("Using cached gold data.")
            return pd.read_csv(file_path)

    try:
        remote_df = download_gold_data()
    except Exception as e:
        if file_path.exists():
            print(f"Warning: live download failed, using cached data. Reason: {e}")
            return pd.read_csv(file_path)
        raise RuntimeError(
            f"Failed to download gold data and no cache is available: {e}"
        ) from e

    if file_path.exists():
        local_df = pd.read_csv(file_path)

        if not local_df.empty and not remote_df.empty:
            local_last_date = str(local_df["Date"].iloc[-1])
            remote_last_date = str(remote_df["Date"].iloc[-1])

            if local_last_date == remote_last_date:
                print("Remote data unchanged. Using existing file.")
                return local_df

    remote_df.to_csv(file_path, index=False)
    print("Downloaded and updated gold data.")

    return remote_df


if __name__ == "__main__":
    df = collect_gold_data()
    print(f"[ingest] available rows: {len(df)}")
    print(f"[ingest] latest date: {df['Date'].iloc[-1]}")
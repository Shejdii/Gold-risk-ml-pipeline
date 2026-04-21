import pandas as pd
import numpy as np
from pathlib import Path


# 1st layer:
# short-term (1 day) returns
def compute_short_term_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return_1d"] = df["Close"].pct_change(1)
    return df


# medium-term (5 day) returns
def compute_medium_term_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return_5d"] = df["Close"].pct_change(5)
    return df


# long-term (21 day) returns
def compute_long_term_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return_21d"] = df["Close"].pct_change(21)
    return df


# longer momentum context
def compute_longer_term_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return_63d"] = df["Close"].pct_change(63)
    return df


# 2nd layer: Volatility features
def compute_short_term_volatility(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vol_7d"] = df["return_1d"].rolling(7).std()
    return df


def compute_long_term_volatility(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vol_21d"] = df["return_1d"].rolling(21).std()
    return df


def compute_long_horizon_volatility_252d(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vol_252d"] = df["return_1d"].rolling(252).std()
    return df


def compute_vol_ratio_7_21(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vol_ratio_7_21"] = df["vol_7d"] / df["vol_21d"]
    df["vol_ratio_7_21"] = df["vol_ratio_7_21"].replace([np.inf, -np.inf], np.nan)
    return df


def ratio_21d_252d(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ratio_21d_252d"] = df["vol_21d"] / df["vol_252d"]
    df["ratio_21d_252d"] = df["ratio_21d_252d"].replace([np.inf, -np.inf], np.nan)
    return df


# trend / mean reversion proxy
def compute_price_vs_ma21(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ma21 = df["Close"].rolling(21).mean()
    df["price_vs_ma21"] = df["Close"] / ma21
    df["price_vs_ma21"] = df["price_vs_ma21"].replace([np.inf, -np.inf], np.nan)
    return df


# 3rd layer: Regime logic from vol 21d quantiles and regime labels
def compute_volatility_regime(df: pd.DataFrame, q33: float, q66: float) -> pd.DataFrame:
    df = df.copy()
    conditions = [
        (df["vol_21d"] <= q33),
        (df["vol_21d"] > q33) & (df["vol_21d"] <= q66),
        (df["vol_21d"] > q66),
    ]
    df["regime_current"] = pd.Series(
        np.select(conditions, ["LOW", "MEDIUM", "HIGH"], default="UNKNOWN"),
        index=df.index,
        dtype="string",
    )
    return df


# 4th layer: Anomaly detection
def compute_volatility_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mean_ = df["vol_7d"].rolling(60).mean()
    std_ = df["vol_7d"].rolling(60).std().replace(0, np.nan)
    df["z_score"] = (df["vol_7d"] - mean_) / std_
    df["anomaly"] = (df["z_score"].abs() > 2).astype(int)
    return df


def anomaly_flag_252d(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mean_ = df["ratio_21d_252d"].rolling(60).mean()
    std_ = df["ratio_21d_252d"].rolling(60).std().replace(0, np.nan)
    df["z_score_ratio"] = (df["ratio_21d_252d"] - mean_) / std_
    df["anomaly_ratio"] = (df["z_score_ratio"].abs() > 2).astype(int)
    return df


# 5th layer: forward risk targets
def compute_future_5d_vol_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["future_5d_vol"] = df["return_1d"].rolling(5).std().shift(-5)
    return df


def build_features_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_short_term_returns(df)
    df = compute_medium_term_returns(df)
    df = compute_long_term_returns(df)
    df = compute_longer_term_returns(df)

    df = compute_short_term_volatility(df)
    df = compute_long_term_volatility(df)
    df = compute_long_horizon_volatility_252d(df)

    df = compute_vol_ratio_7_21(df)
    df = ratio_21d_252d(df)
    df = compute_price_vs_ma21(df)

    df = anomaly_flag_252d(df)

    q33 = df["vol_21d"].quantile(0.33)
    q66 = df["vol_21d"].quantile(0.66)
    df = compute_volatility_regime(df, q33, q66)

    df = compute_volatility_anomaly(df)
    df = compute_future_5d_vol_target(df)

    return df


def build_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runtime-friendly feature generation:
    - no train/val/test split
    - no file writes
    - returns full feature dataframe
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    features_df = build_features_pipeline(df)

    needed = [
        "return_1d",
        "return_5d",
        "return_21d",
        "return_63d",
        "vol_7d",
        "vol_21d",
        "vol_252d",
        "vol_ratio_7_21",
        "ratio_21d_252d",
        "price_vs_ma21",
        "z_score",
    ]
    features_df = features_df.dropna(subset=needed).reset_index(drop=True)

    return features_df


def split_time_series(df: pd.DataFrame, train_size=0.7, val_size=0.15):
    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def add_future_regime(df, q33, q66):
    bins = [-np.inf, q33, q66, np.inf]
    labels = ["LOW", "MEDIUM", "HIGH"]
    df = df.copy()
    df["future_regime"] = pd.cut(df["future_5d_vol"], bins=bins, labels=labels)
    return df


def feature_engineering_main():
    preprocessed_path = Path("data/processed/xauusd_preprocessed.csv")
    if not preprocessed_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data file not found at {preprocessed_path}"
        )

    df = pd.read_csv(preprocessed_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    features_df = build_features_pipeline(df)

    needed = [
        "return_1d",
        "return_5d",
        "return_21d",
        "return_63d",
        "vol_7d",
        "vol_21d",
        "vol_252d",
        "vol_ratio_7_21",
        "ratio_21d_252d",
        "price_vs_ma21",
        "z_score",
        "future_5d_vol",
    ]
    features_df = features_df.dropna(subset=needed).reset_index(drop=True)

    train_df, val_df, test_df = split_time_series(features_df)

    q33 = train_df["future_5d_vol"].quantile(0.33)
    q66 = train_df["future_5d_vol"].quantile(0.66)

    train_df = add_future_regime(train_df, q33, q66)
    val_df = add_future_regime(val_df, q33, q66)
    test_df = add_future_regime(test_df, q33, q66)

    out_dir = Path("data/features")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print(f"[features] saved train/val/test to {out_dir}")
    print(f"[features] q33={q33:.6f} q66={q66:.6f}")


if __name__ == "__main__":
    feature_engineering_main()
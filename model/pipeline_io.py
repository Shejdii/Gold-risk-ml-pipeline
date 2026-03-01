import pandas as pd


def load_features_data(file_path: str, time_col: str | None = None) -> pd.DataFrame:
    if time_col:
        return pd.read_csv(file_path, parse_dates=[time_col])
    return pd.read_csv(file_path)


def get_time_splits(df: pd.DataFrame, time_col: str, train_size=0.7, val_size=0.15):
    if train_size + val_size >= 1:
        raise ValueError("train_size + val_size must be < 1")

    df = df.sort_values(time_col).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    return train_df, val_df, test_df


def get_features_cols(df: pd.DataFrame, drop_cols: list[str]):
    return [c for c in df.columns if c not in drop_cols]


def get_data_range(df: pd.DataFrame, time_col: str):
    return df[time_col].min(), df[time_col].max()

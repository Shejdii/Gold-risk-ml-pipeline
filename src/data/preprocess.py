import pandas as pd
from pathlib import Path

print("Preprocess script started")


def save_preprocessed_data(
    df: pd.DataFrame, output_dir: str = "data/processed"
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "xauusd_preprocessed.csv"
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")


def preprocess_gold_data(df: pd.DataFrame) -> pd.DataFrame:

    # sprawdzenie czy zawiera wymagane kolumny
    required_columns = {"Date", "Open", "High", "Low", "Close"}
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Konwersja kolumny 'Date' na datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # sprawdzenie poprawności typów danych
    numeric_cols = [
        "Open",
        "High",
        "Low",
        "Close",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Usunięcie wierszy z niepoprawnymi datami
    df = df.dropna(subset=["Date"])

    # Usunięcie duplikatów (jeśli istnieją)
    df = df.drop_duplicates(subset=["Date"], keep="last")

    # Usunięcie wierszy z brakującymi wartościami w kluczowych kolumnach
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])

    # Sortowanie danych po dacie
    df = df.sort_values("Date").reset_index(drop=True)

    # sanity: prices > 0
    for c in ["Open", "High", "Low", "Close"]:
        df = df[df[c] > 0]

    # sanity: High >= Low
    df = df[(df["Open"] >= df["Low"]) & (df["Open"] <= df["High"])]
    df = df[(df["Close"] >= df["Low"]) & (df["Close"] <= df["High"])]

    return df.reset_index(drop=True)


if __name__ == "__main__":
    raw_data_path = Path("data/raw/xauusd_latest.csv")
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_data_path}")

    raw_df = pd.read_csv(raw_data_path)
    preprocessed_df = preprocess_gold_data(raw_df)
    save_preprocessed_data(preprocessed_df)

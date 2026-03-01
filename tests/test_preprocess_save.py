import pandas as pd
from pathlib import Path

from src.data.preprocess import save_preprocessed_data


def test_save_preprocessed_data_creates_file(tmp_path):
    # 1. Tworzymy testowy DataFrame
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "Open": [100, 105],
            "High": [110, 115],
            "Low": [90, 95],
            "Close": [105, 110],
        }
    )

    # 2. Używamy tymczasowego katalogu pytest
    output_dir = tmp_path / "processed"

    # 3. Wywołujemy funkcję
    save_preprocessed_data(df, output_dir=str(output_dir))

    # 4. Sprawdzamy czy plik istnieje
    output_file = output_dir / "xauusd_preprocessed.csv"
    assert output_file.exists()

    # 5. Sprawdzamy czy zawartość się zgadza
    saved_df = pd.read_csv(output_file)
    assert len(saved_df) == 2
    assert list(saved_df.columns) == list(df.columns)

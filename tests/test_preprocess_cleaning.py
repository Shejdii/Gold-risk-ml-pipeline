import pandas as pd

from src.data.preprocess import preprocess_gold_data


def test_preprocess_cleans_data_correctly():
    # Dane testowe z błędami
    df = pd.DataFrame(
        {
            "Date": ["2024-01-02", "bad-date", "2024-01-01", "2024-01-01"],
            "Open": [100, 100, -5, 101],
            "High": [110, 110, 105, 90],  # ostatni wiersz: High < Low
            "Low": [90, 90, 95, 95],
            "Close": [105, 105, 100, 100],
        }
    )

    cleaned = preprocess_gold_data(df)

    # 1️ Usunięto bad-date
    assert cleaned["Date"].isna().sum() == 0

    # 2️ Usunięto ujemne ceny
    assert (cleaned["Open"] > 0).all()

    # 3️ Usunięto High < Low
    assert (cleaned["High"] >= cleaned["Low"]).all()

    # 4️ Usunięto duplikaty dat (zostaje 1 rekord na datę)
    assert cleaned["Date"].nunique() == len(cleaned)

    # 5️ Dane są posortowane rosnąco po dacie
    assert cleaned["Date"].is_monotonic_increasing

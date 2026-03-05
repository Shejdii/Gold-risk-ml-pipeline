# train_regime_classifier.py
from pathlib import Path
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_DIR = Path("data/features")
ART_DIR = Path("artifacts/models")
ART_DIR.mkdir(parents=True, exist_ok=True)

TIME_COL = "Date"
TARGET_COL = "future_regime"

DROP_COLS = {
    TIME_COL,
    "Close",  # opcjonalnie usuń surową cenę
    "future_5d_vol",  # target regresji
    "future_regime",  # target klasyfikacji
    "regime_current",  # monitoring / bieżący label
    "anomaly",
    "z_score",  # anomaly features/flags (możesz zostawić jako feature jeśli chcesz)
    "anomaly_ratio",
    "z_score_ratio",
}


def make_X_y(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols]
    y = df[TARGET_COL]
    return X, y, feature_cols


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv", parse_dates=[TIME_COL])
    val_df = pd.read_csv(DATA_DIR / "val.csv", parse_dates=[TIME_COL])

    X_train, y_train, feature_cols = make_X_y(train_df)
    X_val, y_val, _ = make_X_y(val_df)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)

    print(f"[regime] val_accuracy={acc:.4f}")

    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "time_col": TIME_COL,
            "target_col": TARGET_COL,
        },
        ART_DIR / "regime_classifier.pkl",
    )
    print(f"[regime] saved artifacts to {ART_DIR / 'regime_classifier.pkl'}")


if __name__ == "__main__":
    main()

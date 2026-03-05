# train_risk_5d_regressor.py
from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import numpy as np

DATA_DIR = Path("data/features")
ART_DIR = Path("artifacts/models")
ART_DIR.mkdir(parents=True, exist_ok=True)

TIME_COL = "Date"
TARGET_COL = "future_5d_vol"

DROP_COLS = {
    TIME_COL,
    "Close",
    "future_5d_vol",
    "future_regime",
    "regime_current",
}


def make_X_y(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols]
    y = df[TARGET_COL]
    return X, y, feature_cols


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv", parse_dates=[TIME_COL])
    val_df = pd.read_csv(DATA_DIR / "val.csv", parse_dates=[TIME_COL])

    X_train, y_train, feature_cols = make_X_y(train_df)
    X_val, y_val, _ = make_X_y(val_df)

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    score = rmse(y_val, preds)

    print(f"[risk5d] val_rmse={score:.6f}")

    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "time_col": TIME_COL,
            "target_col": TARGET_COL,
        },
        ART_DIR / "risk_5d_regressor.pkl",
    )
    print(f"[risk5d] saved artifacts to {ART_DIR / 'risk_5d_regressor.pkl'}")


if __name__ == "__main__":
    main()

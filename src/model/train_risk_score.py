from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from src.utils.metrics_io import write_metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor

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


def evaluate_model(name, model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    score = rmse(y_val, preds)

    print(f"[{name}] val_rmse={score:.6f}")
    return {"name": name, "model": model, "rmse": score}


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv", parse_dates=[TIME_COL])
    val_df = pd.read_csv(DATA_DIR / "val.csv", parse_dates=[TIME_COL])

    X_train, y_train, feature_cols = make_X_y(train_df)
    X_val, y_val, _ = make_X_y(val_df)

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)

    dummy = DummyRegressor(strategy="mean")

    ridge = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ]
    )

    rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "reg",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=5,
                    random_state=42,
                ),
            ),
        ]
    )

    results = []
    results.append(evaluate_model("dummy_mean", dummy, X_train, y_train, X_val, y_val))
    results.append(evaluate_model("ridge", ridge, X_train, y_train, X_val, y_val))
    results.append(evaluate_model("random_forest", rf, X_train, y_train, X_val, y_val))

    best = min(results[1:], key=lambda r: r["rmse"])
    best_model = best["model"]

    print(f"[risk5d] selected_model={best['name']}")
    print(f"[risk5d] selected_rmse={best['rmse']:.6f}")

    write_metrics(
    "artifacts/metrics/regressor_metrics.json",
    {
        "selected_model_name": best["name"],
        "rmse": float(best["rmse"]),
    },
)
    print("[risk5d] wrote metrics to artifacts/metrics/regressor_metrics.json")

    joblib.dump(
        {
            "model": best_model,
            "feature_cols": feature_cols,
            "time_col": TIME_COL,
            "target_col": TARGET_COL,
            "selected_model_name": best["name"],
        },
        ART_DIR / "risk_5d_regressor.pkl",
    )
    print(f"[risk5d] saved artifacts to {ART_DIR / 'risk_5d_regressor.pkl'}")


if __name__ == "__main__":
    main()
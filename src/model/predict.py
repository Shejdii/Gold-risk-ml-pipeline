from pathlib import Path
import joblib
import pandas as pd
import numpy as np

ART_DIR = Path("artifacts/models")
DATA_DIR = Path("data/features")

REGIME_MODEL_PATH = ART_DIR / "regime_classifier.pkl"
RISK_MODEL_PATH = ART_DIR / "risk_5d_regressor.pkl"

TIME_COL = "Date"


def load_bundle(path: Path):
    bundle = joblib.load(path)
    return bundle["model"], bundle["feature_cols"]


def main():
    test_path = DATA_DIR / "test.csv"
    df = pd.read_csv(test_path, parse_dates=[TIME_COL])

    regime_model, regime_cols = load_bundle(REGIME_MODEL_PATH)
    risk_model, risk_cols = load_bundle(RISK_MODEL_PATH)

    # X for each model (may differ)
    X_regime = df[regime_cols].replace([np.inf, -np.inf], np.nan)
    X_risk = df[risk_cols].replace([np.inf, -np.inf], np.nan)

    pred_regime = regime_model.predict(X_regime)
    pred_risk = risk_model.predict(X_risk)

    out = pd.DataFrame(
        {
            "Date": df[TIME_COL],
            "pred_future_regime": pred_regime,
            "pred_future_5d_vol": pred_risk,
        }
    )

    # optional: include anomaly flags if present
    for c in ["anomaly", "anomaly_ratio"]:
        if c in df.columns:
            out[c] = df[c].values

    out_dir = Path("artifacts/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_predictions.csv"
    out.to_csv(out_path, index=False)

    print(f"[predict] saved -> {out_path}")
    print(out.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()

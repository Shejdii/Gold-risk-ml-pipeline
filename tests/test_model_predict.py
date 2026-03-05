import numpy as np
import pandas as pd
import joblib
from pathlib import Path

FEATURE_DIR = Path("data/features")
MODEL_DIR = Path("artifacts/models")

REGIME_PATH = MODEL_DIR / "regime_classifier.pkl"
RISK_PATH = MODEL_DIR / "risk_5d_regressor.pkl"


def test_models_can_load():
    assert REGIME_PATH.exists(), f"Missing model file: {REGIME_PATH}"
    assert RISK_PATH.exists(), f"Missing model file: {RISK_PATH}"

    regime_bundle = joblib.load(REGIME_PATH)
    risk_bundle = joblib.load(RISK_PATH)

    assert "model" in regime_bundle and "feature_cols" in regime_bundle
    assert "model" in risk_bundle and "feature_cols" in risk_bundle


def test_models_can_predict_on_test_row():
    df = pd.read_csv(FEATURE_DIR / "test.csv")
    assert len(df) > 0, "data/features/test.csv is empty"

    # bierzemy 1 rekord
    row = df.tail(1).copy()

    regime_bundle = joblib.load(REGIME_PATH)
    risk_bundle = joblib.load(RISK_PATH)

    regime_model = regime_bundle["model"]
    regime_cols = regime_bundle["feature_cols"]

    risk_model = risk_bundle["model"]
    risk_cols = risk_bundle["feature_cols"]

    # sanity: wszystkie kolumny istnieją
    missing_regime = set(regime_cols) - set(row.columns)
    missing_risk = set(risk_cols) - set(row.columns)

    assert not missing_regime, f"Missing columns for regime model: {missing_regime}"
    assert not missing_risk, f"Missing columns for risk model: {missing_risk}"

    X_regime = row[regime_cols].replace([np.inf, -np.inf], np.nan)
    X_risk = row[risk_cols].replace([np.inf, -np.inf], np.nan)

    assert not X_regime.isna().any().any(), "NaN in X_regime"
    assert not X_risk.isna().any().any(), "NaN in X_risk"

    pred_regime = regime_model.predict(X_regime)[0]
    pred_risk = float(risk_model.predict(X_risk)[0])

    assert str(pred_regime) in {
        "LOW",
        "MEDIUM",
        "HIGH",
    }, f"Invalid predicted regime: {pred_regime}"
    assert pred_risk >= 0, f"Predicted risk should be >= 0, got {pred_risk}"

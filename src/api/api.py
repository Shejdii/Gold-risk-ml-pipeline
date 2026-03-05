from fastapi import FastAPI
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

app = FastAPI()

# building a simple API to serve predictions from the trained models
regime_bundle = joblib.load("artifacts/models/regime_classifier.pkl")
risk_bundle = joblib.load("artifacts/models/risk_5d_regressor.pkl")

regime_model = regime_bundle["model"]
regime_cols = regime_bundle["feature_cols"]

risk_model = risk_bundle["model"]
risk_cols = risk_bundle["feature_cols"]


@app.get("/predict/latest")
def predict_latest():
    df = pd.read_csv("data/features/test.csv", parse_dates=["Date"])
    last = df.tail(1)

    X_regime = last[regime_cols].replace([np.inf, -np.inf], np.nan)
    X_risk = last[risk_cols].replace([np.inf, -np.inf], np.nan)

    pred_regime = regime_model.predict(X_regime)[0]
    pred_risk = float(risk_model.predict(X_risk)[0])

    return {
        "date": str(last["Date"].values[0]),
        "pred_future_regime": pred_regime,
        "pred_future_5d_vol": pred_risk,
    }


@app.get("/")
def root():
    return {"status": "ok", "service": "gold-risk-api"}


@app.get("/health")
def health():
    return {"status": "ok"}

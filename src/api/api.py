from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "artifacts" / "models"
DATA_PATH = BASE_DIR / "data" / "features" / "test.csv"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Gold Risk API")

regime_bundle = joblib.load(MODEL_DIR / "regime_classifier.pkl")
risk_bundle = joblib.load(MODEL_DIR / "risk_5d_regressor.pkl")

regime_model = regime_bundle["model"]
regime_cols = regime_bundle["feature_cols"]

risk_model = risk_bundle["model"]
risk_cols = risk_bundle["feature_cols"]


def predict_latest_result() -> dict:
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    last = df.tail(1)

    X_regime = last[regime_cols].replace([np.inf, -np.inf], np.nan)
    X_risk = last[risk_cols].replace([np.inf, -np.inf], np.nan)

    pred_regime = regime_model.predict(X_regime)[0]
    pred_risk = float(risk_model.predict(X_risk)[0])

    regime_map = {
        0: "LOW",
        1: "MEDIUM",
        2: "HIGH",
        "LOW": "LOW",
        "MEDIUM": "MEDIUM",
        "HIGH": "HIGH",
    }

    readable_regime = regime_map.get(pred_regime, str(pred_regime))

    return {
        "date": last["Date"].iloc[0].strftime("%Y-%m-%d"),
        "pred_future_regime": readable_regime,
        "pred_future_5d_vol": round(pred_risk, 6),
    }


@app.get("/predict/latest")
def predict_latest():
    try:
        return predict_latest_result()
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
        )


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
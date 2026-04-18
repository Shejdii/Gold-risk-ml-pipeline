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

REGIME_MODEL_PATH = MODEL_DIR / "regime_classifier.pkl"
RISK_MODEL_PATH = MODEL_DIR / "risk_5d_regressor.pkl"

app = FastAPI(title="Gold Risk API")


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required {label} file not found: {path}")


def load_regime_bundle() -> dict:
    require_file(REGIME_MODEL_PATH, "regime model")
    return joblib.load(REGIME_MODEL_PATH)


def load_risk_bundle() -> dict:
    require_file(RISK_MODEL_PATH, "risk model")
    return joblib.load(RISK_MODEL_PATH)


def load_feature_data() -> pd.DataFrame:
    require_file(DATA_PATH, "feature data")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

    if df.empty:
        raise ValueError("Feature data file exists but is empty.")

    return df


def predict_latest_result() -> dict:
    regime_bundle = load_regime_bundle()
    risk_bundle = load_risk_bundle()

    regime_model = regime_bundle["model"]
    regime_cols = regime_bundle["feature_cols"]

    risk_model = risk_bundle["model"]
    risk_cols = risk_bundle["feature_cols"]

    df = load_feature_data()
    last = df.tail(1)

    X_regime = last[regime_cols].replace([np.inf, -np.inf], np.nan)
    X_risk = last[risk_cols].replace([np.inf, -np.inf], np.nan)

    if X_regime.isna().any().any():
        raise ValueError("Latest regime feature row contains NaN or infinite values.")

    if X_risk.isna().any().any():
        raise ValueError("Latest risk feature row contains NaN or infinite values.")

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
        "status": "ok",
        "prediction": {
            "date": last["Date"].iloc[0].strftime("%Y-%m-%d"),
            "future_regime": readable_regime,
            "future_5d_vol": round(pred_risk, 6),
        },
    }


@app.get("/predict/latest")
def predict_latest():
    try:
        return predict_latest_result()
    except FileNotFoundError as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(exc)},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Unexpected server error: {exc}"},
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "files": {
            "regime_model": REGIME_MODEL_PATH.exists(),
            "risk_model": RISK_MODEL_PATH.exists(),
            "feature_data": DATA_PATH.exists(),
            "static_dir": STATIC_DIR.exists(),
        },
    }


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
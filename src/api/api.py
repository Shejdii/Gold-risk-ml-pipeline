from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.data.ingest import collect_gold_data
from src.data.preprocess import preprocess_gold_data
from src.features.build_features import build_features_df

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "artifacts" / "models"
STATIC_DIR = BASE_DIR / "static"

REGIME_MODEL_PATH = MODEL_DIR / "regime_classifier.pkl"
RISK_MODEL_PATH = MODEL_DIR / "risk_5d_regressor.pkl"

app = FastAPI(
    title="Gold Risk API",
    description="Inference API for gold volatility regime classification and 5-day forward risk prediction.",
    version="1.0.0",
)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required {label} file not found: {path}")


def load_regime_bundle() -> dict:
    require_file(REGIME_MODEL_PATH, "regime model")
    return joblib.load(REGIME_MODEL_PATH)


def load_risk_bundle() -> dict:
    require_file(RISK_MODEL_PATH, "risk model")
    return joblib.load(RISK_MODEL_PATH)


def build_latest_feature_row() -> pd.DataFrame:
    raw_df = collect_gold_data()
    processed_df = preprocess_gold_data(raw_df)
    features_df = build_features_df(processed_df)

    if features_df.empty:
        raise ValueError("Feature dataframe is empty after runtime feature generation.")

    latest = features_df.tail(1).copy()
    if latest.empty:
        raise ValueError("No latest feature row available.")

    return latest


def predict_latest_result() -> dict:
    regime_bundle = load_regime_bundle()
    risk_bundle = load_risk_bundle()

    regime_model = regime_bundle["model"]
    regime_cols = regime_bundle["feature_cols"]

    risk_model = risk_bundle["model"]
    risk_cols = risk_bundle["feature_cols"]

    last = build_latest_feature_row()

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
    risk_pct = round(pred_risk * 100, 2)

    regime_explanations = {
        "LOW": "Lower expected market turbulence over the next 5 trading days.",
        "MEDIUM": "Moderate expected market turbulence over the next 5 trading days.",
        "HIGH": "Elevated expected market turbulence over the next 5 trading days.",
    }

    return {
        "status": "ok",
        "prediction": {
            "date": last["Date"].iloc[0].strftime("%Y-%m-%d"),
            "market_regime_label": readable_regime,
            "predicted_5d_volatility_pct": risk_pct,
            "regime_explanation": regime_explanations.get(
                readable_regime,
                "Model-generated market regime classification."
            ),
            "volatility_explanation": (
                "Predicted 5-day volatility is a forward-looking estimate of market risk. "
                "Higher values indicate a less stable market."
            ),
            "data_refresh_mode": "runtime_fresh",
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
    ready = REGIME_MODEL_PATH.exists() and RISK_MODEL_PATH.exists()

    return {
        "status": "ok",
        "ready": ready,
        "files": {
            "regime_model": REGIME_MODEL_PATH.exists(),
            "risk_model": RISK_MODEL_PATH.exists(),
            "static_dir": STATIC_DIR.exists(),
        },
    }


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
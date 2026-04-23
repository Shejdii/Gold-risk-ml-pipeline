import json
import sys
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "regime_classifier.pkl"
INPUT_PATH = PROJECT_ROOT / "data" / "features" / "test.csv"


def fail(message: str) -> None:
    print(json.dumps({"success": False, "message": message}, indent=2))
    sys.exit(1)


def main() -> None:
    if not MODEL_PATH.exists():
        fail(f"Model artifact not found: {MODEL_PATH}")

    if not INPUT_PATH.exists():
        fail(f"Smoke input CSV not found: {INPUT_PATH}")

    try:
        artifact = joblib.load(MODEL_PATH)
    except Exception as exc:
        fail(f"Failed to load model artifact: {exc}")

    if not isinstance(artifact, dict):
        fail("Loaded classifier artifact is not a dict as expected.")

    model = artifact.get("model")
    feature_cols = artifact.get("feature_cols")

    if model is None:
        fail("Loaded artifact does not contain 'model'.")

    if not feature_cols:
        fail("Loaded artifact does not contain 'feature_cols'.")

    try:
        df = pd.read_csv(INPUT_PATH)
    except Exception as exc:
        fail(f"Failed to load smoke input CSV: {exc}")

    if df.empty:
        fail("Smoke input CSV is empty.")

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        fail(f"Smoke input CSV is missing required feature columns: {missing_cols[:10]}")

    sample = df[feature_cols].head(1)

    try:
        prediction = model.predict(sample)
    except Exception as exc:
        fail(f"Prediction failed: {exc}")

    prediction_preview = (
        prediction.tolist() if hasattr(prediction, "tolist") else str(prediction)
    )

    print(
        json.dumps(
            {
                "success": True,
                "message": "Smoke prediction passed.",
                "rows_used": len(sample),
                "prediction_preview": prediction_preview,
            },
            indent=2,
        )
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
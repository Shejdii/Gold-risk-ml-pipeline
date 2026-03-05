from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"

RAW_FILE = RAW_DIR / "xauusd_latest.csv"
PREPROCESSED_FILE = PROCESSED_DIR / "xauusd_preprocessed.csv"

TRAIN_FILE = FEATURES_DIR / "train.csv"
VAL_FILE = FEATURES_DIR / "val.csv"
TEST_FILE = FEATURES_DIR / "test.csv"

REGIME_MODEL_FILE = MODELS_DIR / "regime_classifier.pkl"
RISK_MODEL_FILE = MODELS_DIR / "risk_5d_regressor.pkl"

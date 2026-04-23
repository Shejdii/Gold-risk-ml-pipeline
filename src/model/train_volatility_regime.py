from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from src.utils.metrics_io import write_metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.dummy import DummyClassifier

DATA_DIR = Path("data/features")
ART_DIR = Path("artifacts/models")
ART_DIR.mkdir(parents=True, exist_ok=True)

TIME_COL = "Date"
TARGET_COL = "future_regime"

DROP_COLS = {
    TIME_COL,
    "Close",
    "future_5d_vol",
    "future_regime",
    "regime_current",
    # anomaly features zostawiamy jako potencjalnie użyteczne
}


def make_X_y(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    return X, y, feature_cols


def evaluate_model(name, model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    acc = accuracy_score(y_val, preds)
    bal_acc = balanced_accuracy_score(y_val, preds)
    macro_f1 = f1_score(y_val, preds, average="macro")

    print(f"\n[{name}]")
    print(f"accuracy={acc:.4f}")
    print(f"balanced_accuracy={bal_acc:.4f}")
    print(f"macro_f1={macro_f1:.4f}")
    print(classification_report(y_val, preds))

    return {
        "name": name,
        "model": model,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
    }


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv", parse_dates=[TIME_COL])
    val_df = pd.read_csv(DATA_DIR / "val.csv", parse_dates=[TIME_COL])

    X_train, y_train, feature_cols = make_X_y(train_df)
    X_val, y_val, _ = make_X_y(val_df)

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)

    print("[regime] class distribution train:")
    print(y_train.value_counts(normalize=True).sort_index())
    print("[regime] class distribution val:")
    print(y_val.value_counts(normalize=True).sort_index())

    dummy = DummyClassifier(strategy="most_frequent")
    logreg = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )

    rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=5,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    results = []
    results.append(evaluate_model("dummy_most_frequent", dummy, X_train, y_train, X_val, y_val))
    results.append(evaluate_model("logistic_regression", logreg, X_train, y_train, X_val, y_val))
    results.append(evaluate_model("random_forest", rf, X_train, y_train, X_val, y_val))

    # wybór najlepszego modelu po macro_f1
    best = max(results[1:], key=lambda r: r["macro_f1"])  # pomijamy dummy przy wyborze
    best_model = best["model"]

    print(f"\n[regime] selected_model={best['name']}")
    print(f"[regime] selected_macro_f1={best['macro_f1']:.4f}")
    print(f"[regime] selected_accuracy={best['accuracy']:.4f}")

    write_metrics(
    "artifacts/metrics/classifier_metrics.json",
    {
        "selected_model_name": best["name"],
        "macro_f1": float(best["macro_f1"]),
        "accuracy": float(best["accuracy"]),
        "balanced_accuracy": float(best["balanced_accuracy"]),
    },
)
    print("[regime] wrote metrics to artifacts/metrics/classifier_metrics.json")

    joblib.dump(
        {
            "model": best_model,
            "feature_cols": feature_cols,
            "time_col": TIME_COL,
            "target_col": TARGET_COL,
            "selected_model_name": best["name"],
        },
        ART_DIR / "regime_classifier.pkl",
    )
    print(f"[regime] saved artifacts to {ART_DIR / 'regime_classifier.pkl'}")


if __name__ == "__main__":
    main()
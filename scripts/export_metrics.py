import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"

CLASSIFIER_METRICS_PATH = METRICS_DIR / "classifier_metrics.json"
REGRESSOR_METRICS_PATH = METRICS_DIR / "regressor_metrics.json"
FINAL_METRICS_PATH = METRICS_DIR / "gold_metrics.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    combined: dict = {}

    if CLASSIFIER_METRICS_PATH.exists():
        combined.update(load_json(CLASSIFIER_METRICS_PATH))

    if REGRESSOR_METRICS_PATH.exists():
        combined.update(load_json(REGRESSOR_METRICS_PATH))

    if not combined:
        raise FileNotFoundError(
            "No intermediate metrics files found. Expected classifier_metrics.json and/or regressor_metrics.json."
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with FINAL_METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(combined, file, indent=2)

    print(f"[metrics] wrote combined metrics to {FINAL_METRICS_PATH}")


if __name__ == "__main__":
    main()
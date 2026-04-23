from fastapi.testclient import TestClient
from src.api.api import app


def test_health_endpoint():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200

    data = r.json()
    assert data["status"] == "ok"
    assert "files" in data


def test_predict_latest_endpoint_exists():
    client = TestClient(app)
    r = client.get("/predict/latest")
    assert r.status_code == 200

    data = r.json()
    assert data["status"] == "ok"
    assert "prediction" in data

    pred = data["prediction"]
    assert "date" in pred
    assert "market_regime_label" in pred
    assert "predicted_5d_volatility_pct" in pred
    assert "regime_explanation" in pred
    assert "volatility_explanation" in pred
    assert "data_refresh_mode" in pred

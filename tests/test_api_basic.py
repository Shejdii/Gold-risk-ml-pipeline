from fastapi.testclient import TestClient
from src.api.api import app


def test_health_endpoint_exists():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200


def test_predict_latest_endpoint_exists():
    client = TestClient(app)
    r = client.get("/predict/latest")
    assert r.status_code == 200

    data = r.json()
    assert "date" in data
    assert "pred_future_regime" in data
    assert "pred_future_5d_vol" in data

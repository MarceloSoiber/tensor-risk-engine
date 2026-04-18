from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_returns_backend_online() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Backend online"}


def test_health_returns_ok() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_risk_result() -> None:
    payload = {
        "amount": 250.0,
        "velocity_1h": 2,
        "merchant_risk": 0.2,
        "device_trust": 0.9,
    }

    response = client.post("/api/v1/predict", json=payload)
    body = response.json()

    assert response.status_code == 200
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["decision"] in {"approve", "review", "reject"}
    assert isinstance(body["reasons"], list)
    assert body["model_version"] == "heuristic-v1"


def test_predict_rejects_invalid_payload() -> None:
    payload = {
        "amount": -10.0,
        "velocity_1h": 2,
        "merchant_risk": 0.2,
        "device_trust": 0.9,
    }

    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 422

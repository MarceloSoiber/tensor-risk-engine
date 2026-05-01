import pytest
from pydantic import ValidationError

from app.controllers.v1.predict_controller import health, predict
from app.main import app
from app.main import root
from app.schemas.request import PredictRequest


def test_core_routes_are_registered() -> None:
    routes = {(route.path, tuple(sorted(route.methods or []))) for route in app.routes}

    assert ("/", ("GET",)) in routes
    assert ("/api/v1/health", ("GET",)) in routes
    assert ("/api/v1/predict", ("POST",)) in routes


def test_root_returns_backend_online() -> None:
    assert root() == {"message": "Backend online"}


def test_health_returns_ok() -> None:
    assert health() == {"status": "ok"}


def test_predict_returns_risk_result() -> None:
    response = predict(
        PredictRequest(
            amount=250.0,
            velocity_1h=2,
            merchant_risk=0.2,
            device_trust=0.9,
        ),
    )
    body = response.model_dump(mode="json")

    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["decision"] in {"approve", "review", "reject"}
    assert isinstance(body["reasons"], list)
    assert body["model_version"] == "heuristic-v1"


def test_predict_rejects_invalid_payload() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(
            amount=-10.0,
            velocity_1h=2,
            merchant_risk=0.2,
            device_trust=0.9,
        )

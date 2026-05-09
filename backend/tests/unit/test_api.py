import asyncio

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.controllers.v1.predict_controller import health, predict
from app.main import app
from app.main import root
from app.schemas.request import PredictRequest


def _predict_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "amount": 250.0,
        "transaction_datetime": "2020-06-21T03:15:00",
        "merchant": "fraud_Kilback LLC",
        "category": "grocery_pos",
        "gender": "F",
        "state": "NY",
        "job": "Engineer",
        "city_population": 125000,
        "customer_latitude": 40.7128,
        "customer_longitude": -74.0060,
        "merchant_latitude": 40.7306,
        "merchant_longitude": -73.9352,
        "transactions_last_hour": 2,
        "transactions_last_24h": 8,
        "average_amount_24h": 75.0,
    }
    payload.update(overrides)
    return payload


def test_core_routes_are_registered() -> None:
    routes = {(route.path, tuple(sorted(route.methods or []))) for route in app.routes}

    assert ("/", ("GET",)) in routes
    assert ("/api/v1/health", ("GET",)) in routes
    assert ("/api/v1/predict", ("POST",)) in routes
    assert ("/api/v1/dashboard/overview", ("GET",)) in routes
    assert ("/api/v1/transactions", ("GET",)) in routes


def test_root_returns_backend_online() -> None:
    assert asyncio.run(root()) == {"message": "Backend online"}


def test_health_returns_ok() -> None:
    assert asyncio.run(health()) == {"status": "ok"}


def test_predict_returns_risk_result() -> None:
    response = asyncio.run(predict(PredictRequest(**_predict_payload())))
    body = response.model_dump(mode="json")

    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["decision"] in {"approve", "review", "reject"}
    assert isinstance(body["reasons"], list)
    assert body["model_version"].startswith(("baseline:", "heuristic-v1"))


def test_predict_rejects_invalid_training_job_id() -> None:
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            predict(
                PredictRequest(**_predict_payload(training_job_id="not-a-real-training-job")),
            ),
        )
    assert exc_info.value.status_code == 422


def test_predict_rejects_invalid_payload() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(**_predict_payload(amount=-10.0))


def test_predict_rejects_removed_risk_inputs() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(**_predict_payload(merchant_risk=0.2, device_trust=0.9))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("category", "unsupported_category"),
        ("gender", "X"),
        ("state", "New York"),
        ("transactions_last_hour", 9),
    ],
)
def test_predict_rejects_invalid_real_transaction_fields(field: str, value: object) -> None:
    overrides = {field: value}
    if field == "transactions_last_hour":
        overrides["transactions_last_24h"] = 8

    with pytest.raises(ValidationError):
        PredictRequest(**_predict_payload(**overrides))

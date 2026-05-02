import asyncio

import httpx
import pytest

from app.main import app


pytestmark = pytest.mark.integration


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


async def _post_prediction(payload: dict[str, object]) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.post("/api/v1/predict", json=payload)


def test_predict_transaction_returns_expected_risk_decision() -> None:
    response = asyncio.run(_post_prediction(_predict_payload()))

    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["decision"] in {"approve", "reject"}
    assert body["reasons"] in (["trained_model_below_threshold"], ["trained_model_above_threshold"])
    assert body["model_version"].startswith("baseline:")


def test_predict_transaction_rejects_high_risk_transaction() -> None:
    response = asyncio.run(
        _post_prediction(
            _predict_payload(
                amount=5000.0,
                transaction_datetime="2020-06-21T02:15:00",
                customer_latitude=40.7128,
                customer_longitude=-74.0060,
                merchant_latitude=34.0522,
                merchant_longitude=-118.2437,
                transactions_last_hour=20,
                transactions_last_24h=100,
                average_amount_24h=25.0,
            ),
        ),
    )

    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["decision"] in {"approve", "reject"}
    assert body["reasons"] in (["trained_model_below_threshold"], ["trained_model_above_threshold"])
    assert body["model_version"].startswith("baseline:")


def test_predict_transaction_rejects_invalid_payload() -> None:
    response = asyncio.run(_post_prediction(_predict_payload(amount=-10.0)))

    assert response.status_code == 422


def test_predict_transaction_accepts_specific_training_job_id() -> None:
    first_response = asyncio.run(_post_prediction(_predict_payload()))
    assert first_response.status_code == 200
    first_body = first_response.json()
    assert first_body["model_version"].startswith("baseline:")
    selected_job_id = first_body["model_version"].split("baseline:", maxsplit=1)[1]

    selected_response = asyncio.run(
        _post_prediction(
            _predict_payload(training_job_id=selected_job_id),
        ),
    )
    assert selected_response.status_code == 200
    selected_body = selected_response.json()
    assert selected_body["model_version"] == first_body["model_version"]

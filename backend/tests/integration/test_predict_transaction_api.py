import asyncio

import httpx
import pytest

from app.main import app


pytestmark = pytest.mark.integration


async def _post_prediction(payload: dict[str, float | int]) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.post("/api/v1/predict", json=payload)


def test_predict_transaction_returns_expected_risk_decision() -> None:
    response = asyncio.run(
        _post_prediction(
            {
                "amount": 250.0,
                "velocity_1h": 2,
                "merchant_risk": 0.2,
                "device_trust": 0.9,
            },
        ),
    )

    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["decision"] in {"approve", "reject"}
    assert body["reasons"] in (["trained_model_below_threshold"], ["trained_model_above_threshold"])
    assert body["model_version"].startswith("baseline:")


def test_predict_transaction_rejects_high_risk_transaction() -> None:
    response = asyncio.run(
        _post_prediction(
            {
                "amount": 5000.0,
                "velocity_1h": 20,
                "merchant_risk": 1.0,
                "device_trust": 0.0,
            },
        ),
    )

    assert response.status_code == 200

    body = response.json()
    assert body["risk_score"] == pytest.approx(1.0)
    assert body["decision"] == "reject"
    assert body["reasons"] == ["trained_model_above_threshold"]
    assert body["model_version"].startswith("baseline:")


def test_predict_transaction_rejects_invalid_payload() -> None:
    response = asyncio.run(
        _post_prediction(
            {
                "amount": -10.0,
                "velocity_1h": 2,
                "merchant_risk": 0.2,
                "device_trust": 0.9,
            },
        ),
    )

    assert response.status_code == 422

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.controllers.v1 import ai_analysis_controller
from app.llm.langchain_client import AiAnalysisProviderError, LangChainLocalLlmClient, OpenRouterLlmClient
from app.main import app
from app.schemas.ai_analysis import AiAnalysisDataSummary, AiAnalysisFilters, AiAnalysisResponse


class FakeAiAnalysisService:
    def __init__(self) -> None:
        self.received_question: str | None = None
        self.received_filters: AiAnalysisFilters | None = None
        self.deleted_analysis_id: int | None = None

    def analyze(self, *, question: str, filters: AiAnalysisFilters) -> AiAnalysisResponse:
        self.received_question = question
        self.received_filters = filters
        return AiAnalysisResponse(
            analysis_id=7,
            question=question,
            answer="Review elevated risk transactions.",
            insights=["One transaction has a high risk score."],
            recommended_actions=["Escalate the transaction for analyst review."],
            data_summary=AiAnalysisDataSummary(
                transaction_count=1,
                filtered_total=1,
                source=filters.source,
                decision=filters.decision,
                category=filters.category,
                average_risk_score=0.8,
                total_amount=250.0,
            ),
            transactions=[],
            model="local-test",
            created_at=datetime(2024, 1, 2, 3, 4, 5),
        )

    def list_history(self, *, limit: int, offset: int) -> tuple[int, list[AiAnalysisResponse]]:
        return 0, []

    def delete_history_item(self, *, analysis_id: int) -> bool:
        self.deleted_analysis_id = analysis_id
        return analysis_id == 7


class FailingAiAnalysisService(FakeAiAnalysisService):
    def analyze(self, *, question: str, filters: AiAnalysisFilters) -> AiAnalysisResponse:  # noqa: ARG002
        raise AiAnalysisProviderError("offline")


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_ai_analysis_query_endpoint_returns_structured_response(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    service = FakeAiAnalysisService()
    monkeypatch.setattr(ai_analysis_controller, "ai_analysis_service", service)

    response = client.post(
        "/api/v1/ai-analysis/query",
        json={
            "question": "Which transactions need attention?",
            "filters": {
                "source": "analyzed",
                "decision": "review",
                "category": "grocery_pos",
                "limit": 5,
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["analysis_id"] == 7
    assert body["data_summary"]["source"] == "analyzed"
    assert service.received_question == "Which transactions need attention?"
    assert service.received_filters is not None
    assert service.received_filters.limit == 5


def test_ai_analysis_query_endpoint_maps_llm_unavailable_to_503(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ai_analysis_controller, "ai_analysis_service", FailingAiAnalysisService())

    response = client.post(
        "/api/v1/ai-analysis/query",
        json={"question": "Which transactions need attention?"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "AI analysis is unavailable."


def test_ai_analysis_history_endpoint_returns_empty_history(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ai_analysis_controller, "ai_analysis_service", FakeAiAnalysisService())

    response = client.get("/api/v1/ai-analysis/history?limit=10&offset=0")

    assert response.status_code == 200
    assert response.json() == {"total": 0, "limit": 10, "offset": 0, "analyses": []}


def test_ai_analysis_history_delete_endpoint_removes_saved_analysis(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeAiAnalysisService()
    monkeypatch.setattr(ai_analysis_controller, "ai_analysis_service", service)

    response = client.delete("/api/v1/ai-analysis/history/7")

    assert response.status_code == 204
    assert response.content == b""
    assert service.deleted_analysis_id == 7


def test_ai_analysis_history_delete_endpoint_returns_404_when_missing(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ai_analysis_controller, "ai_analysis_service", FakeAiAnalysisService())

    response = client.delete("/api/v1/ai-analysis/history/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "AI analysis history item was not found."


def test_ai_analysis_llm_client_uses_local_when_openrouter_is_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ai_analysis_controller,
        "settings",
        replace(ai_analysis_controller.settings, openrouter_api_key=None, openrouter_model=None),
    )

    llm_client, model = ai_analysis_controller._build_ai_analysis_llm_client()

    assert isinstance(llm_client, LangChainLocalLlmClient)
    assert model == ai_analysis_controller.settings.local_llm_model


def test_ai_analysis_llm_client_uses_openrouter_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ai_analysis_controller,
        "settings",
        replace(ai_analysis_controller.settings, openrouter_api_key="secret-key", openrouter_model="provider/model"),
    )

    llm_client, model = ai_analysis_controller._build_ai_analysis_llm_client()

    assert isinstance(llm_client, OpenRouterLlmClient)
    assert model == "provider/model"

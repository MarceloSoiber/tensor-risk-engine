from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from app.llm.langchain_client import AiAnalysisProviderError
from app.schemas.ai_analysis import AiAnalysisFilters, AiAnalysisQueryRequest
from app.services.ai_analysis_service import AiAnalysisService


class FakeAiAnalysisRepository:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.saved: list[dict[str, object]] = []

    def list_transactions_for_analysis(self, filters: AiAnalysisFilters) -> tuple[int, list[dict[str, object]]]:
        return len(self.rows), self.rows[: filters.limit]

    def save_analysis(self, **kwargs: object) -> dict[str, object]:
        self.saved.append(kwargs)
        return {"id": len(self.saved), "created_at": datetime(2024, 1, 2, 3, 4, 5)}

    def list_analysis_history(self, *, limit: int, offset: int) -> tuple[int, list[dict[str, object]]]:
        rows = [
            {
                "id": index + 1,
                "question": str(item["question"]),
                "answer": str(item["answer"]),
                "insights": item["insights"],
                "recommended_actions": item["recommended_actions"],
                "data_summary": item["data_summary"],
                "model": item["model"],
                "status": item["status"],
                "error_message": item.get("error_message"),
                "created_at": datetime(2024, 1, 2, 3, 4, 5),
            }
            for index, item in enumerate(self.saved)
        ]
        return len(rows), rows[offset : offset + limit]


class FakeLlmClient:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


class FailingLlmClient:
    def generate(self, prompt: str) -> str:  # noqa: ARG002
        raise AiAnalysisProviderError("LLM offline")


def test_ai_analysis_request_strips_question() -> None:
    request = AiAnalysisQueryRequest(question="  Review risky transactions  ")

    assert request.question == "Review risky transactions"
    assert request.filters.source == "all"


def test_ai_analysis_request_rejects_short_question() -> None:
    with pytest.raises(ValidationError):
        AiAnalysisQueryRequest(question="no")


def test_ai_analysis_service_builds_prompt_and_persists_structured_response() -> None:
    repository = FakeAiAnalysisRepository(
        [
            {
                "id": 42,
                "transaction_datetime": datetime(2024, 1, 1, 12, 0, 0),
                "merchant": "Northside Market",
                "category": "grocery_pos",
                "amount": 84.35,
                "risk_score": 0.82,
                "decision": "review",
                "is_fraud": None,
                "model_version": "heuristic-v1",
            }
        ]
    )
    llm = FakeLlmClient(
        '{"answer":"Review high-risk grocery transaction.","insights":["Risk score is elevated."],"recommended_actions":["Check merchant history."]}'
    )
    service = AiAnalysisService(repository=repository, llm_client=llm, model="local-test", max_transactions=25)

    response = service.analyze(question="What should the analyst inspect?", filters=AiAnalysisFilters())

    assert response.analysis_id == 1
    assert response.answer == "Review high-risk grocery transaction."
    assert response.insights == ["Risk score is elevated."]
    assert response.recommended_actions == ["Check merchant history."]
    assert response.data_summary.transaction_count == 1
    assert repository.saved[0]["status"] == "completed"
    assert "Use only the transaction context below" in llm.prompts[0]


def test_ai_analysis_service_persists_failed_provider_attempt() -> None:
    repository = FakeAiAnalysisRepository(
        [
            {
                "id": 99,
                "transaction_datetime": datetime(2024, 1, 1, 12, 0, 0),
                "merchant": "Airport Hotel",
                "category": "travel",
                "amount": 917.45,
            }
        ]
    )
    service = AiAnalysisService(repository=repository, llm_client=FailingLlmClient(), model="local-test", max_transactions=25)

    with pytest.raises(AiAnalysisProviderError):
        service.analyze(question="What is risky?", filters=AiAnalysisFilters())

    assert repository.saved[0]["status"] == "failed"
    assert repository.saved[0]["error_message"] == "LLM offline"


def test_ai_analysis_service_returns_empty_context_without_llm_call() -> None:
    repository = FakeAiAnalysisRepository([])
    llm = FakeLlmClient("{}")
    service = AiAnalysisService(repository=repository, llm_client=llm, model="local-test", max_transactions=25)

    response = service.analyze(question="What is risky?", filters=AiAnalysisFilters())

    assert response.data_summary.transaction_count == 0
    assert response.insights == ["No transaction context was available for the requested filters."]
    assert llm.prompts == []


def test_ai_analysis_service_parses_lm_studio_channel_markers() -> None:
    repository = FakeAiAnalysisRepository(
        [
            {
                "id": 42,
                "transaction_datetime": datetime(2024, 1, 1, 12, 0, 0),
                "merchant": "Northside Market",
                "category": "grocery_pos",
                "amount": 84.35,
            }
        ]
    )
    llm = FakeLlmClient(
        '<|channel|>final <|constrain|>JSON<|message|>{"answer":"Use analyst review.","insights":["Context is limited."],"recommended_actions":["Inspect the transaction."]}'
    )
    service = AiAnalysisService(repository=repository, llm_client=llm, model="local-test", max_transactions=25)

    response = service.analyze(question="What is risky?", filters=AiAnalysisFilters())

    assert response.answer == "Use analyst review."
    assert response.insights == ["Context is limited."]
    assert response.recommended_actions == ["Inspect the transaction."]

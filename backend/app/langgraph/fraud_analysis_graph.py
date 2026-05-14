from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.core.config import settings
from app.llm.langchain_client import AiAnalysisLlmClient, LangChainLocalLlmClient, OpenRouterLlmClient
from app.repositories.ai_analysis_repository import PostgresAiAnalysisRepository
from app.schemas.ai_analysis import AiAnalysisFilters
from app.services.ai_analysis_service import AiAnalysisService


class FraudAnalysisState(TypedDict, total=False):
    question: str
    filters: dict[str, Any]
    normalized_filters: dict[str, Any]
    result: dict[str, Any]


def normalize_request(state: FraudAnalysisState) -> FraudAnalysisState:
    question = str(state.get("question") or "").strip()
    if not question:
        question = "Which transactions deserve analyst attention?"

    filters = AiAnalysisFilters.model_validate(state.get("filters") or {})
    return {
        "question": question,
        "normalized_filters": filters.model_dump(mode="json"),
    }


def run_ai_analysis(state: FraudAnalysisState) -> FraudAnalysisState:
    filters = AiAnalysisFilters.model_validate(state.get("normalized_filters") or {})
    service = AiAnalysisService(
        repository=PostgresAiAnalysisRepository(settings.database_url),
        llm_client=_build_llm_client(),
        model=_resolve_model_name(),
        max_transactions=settings.ai_analysis_max_transactions,
    )
    response = service.analyze(question=str(state["question"]), filters=filters)
    return {"result": response.model_dump(mode="json")}


def _build_llm_client() -> AiAnalysisLlmClient:
    if settings.openrouter_api_key and settings.openrouter_model:
        return OpenRouterLlmClient(
            base_url=settings.openrouter_base_url,
            model=settings.openrouter_model,
            api_key=settings.openrouter_api_key,
        )

    return LangChainLocalLlmClient(
        base_url=settings.local_llm_base_url,
        model=settings.local_llm_model,
        api_key=settings.local_llm_api_key,
    )


def _resolve_model_name() -> str:
    if settings.openrouter_api_key and settings.openrouter_model:
        return settings.openrouter_model
    return settings.local_llm_model


workflow = StateGraph(FraudAnalysisState)
workflow.add_node("normalize_request", normalize_request)
workflow.add_node("run_ai_analysis", run_ai_analysis)
workflow.add_edge(START, "normalize_request")
workflow.add_edge("normalize_request", "run_ai_analysis")
workflow.add_edge("run_ai_analysis", END)

graph = workflow.compile()

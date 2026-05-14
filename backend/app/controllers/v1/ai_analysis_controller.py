from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.core.config import settings
from app.llm.langchain_client import (
    AiAnalysisLlmClient,
    AiAnalysisProviderError,
    LangChainLocalLlmClient,
    OpenRouterLlmClient,
)
from app.repositories.ai_analysis_repository import AiAnalysisPersistenceError, PostgresAiAnalysisRepository
from app.schemas.ai_analysis import (
    AiAnalysisHistoryResponse,
    AiAnalysisObservabilityResponse,
    AiAnalysisQueryRequest,
    AiAnalysisResponse,
)
from app.services.ai_analysis_service import AiAnalysisService

router = APIRouter(prefix="/v1/ai-analysis", tags=["ai-analysis"])


def _build_ai_analysis_llm_client() -> tuple[AiAnalysisLlmClient, str]:
    if settings.openrouter_api_key and settings.openrouter_model:
        return (
            OpenRouterLlmClient(
                base_url=settings.openrouter_base_url,
                model=settings.openrouter_model,
                api_key=settings.openrouter_api_key,
            ),
            settings.openrouter_model,
        )

    return (
        LangChainLocalLlmClient(
            base_url=settings.local_llm_base_url,
            model=settings.local_llm_model,
            api_key=settings.local_llm_api_key,
        ),
        settings.local_llm_model,
    )


llm_client, ai_analysis_model = _build_ai_analysis_llm_client()

ai_analysis_service = AiAnalysisService(
    repository=PostgresAiAnalysisRepository(settings.database_url),
    llm_client=llm_client,
    model=ai_analysis_model,
    max_transactions=settings.ai_analysis_max_transactions,
)


@router.post("/query", response_model=AiAnalysisResponse)
def query_ai_analysis(payload: AiAnalysisQueryRequest) -> AiAnalysisResponse:
    try:
        return ai_analysis_service.analyze(question=payload.question, filters=payload.filters)
    except AiAnalysisProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI analysis is unavailable.",
        ) from exc
    except AiAnalysisPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI analysis data is unavailable.",
        ) from exc


@router.get("/observability", response_model=AiAnalysisObservabilityResponse)
def get_ai_analysis_observability() -> AiAnalysisObservabilityResponse:
    traceable_provider = not settings.local_llm_base_url.rstrip("/").endswith("/api/v1/chat")
    provider_note = (
        "LangSmith traces are emitted for LangChain ChatOpenAI compatible providers."
        if traceable_provider
        else "Current LM Studio /api/v1/chat mode is preserved for the frontend but is not traced by LangSmith."
    )
    return AiAnalysisObservabilityResponse(
        tracing_enabled=settings.langsmith_tracing and bool(settings.langsmith_api_key),
        project=settings.langsmith_project,
        dashboard_url=f"https://smith.langchain.com/o/default/projects/p/{settings.langsmith_project}",
        traceable_provider=traceable_provider,
        provider_note=provider_note,
    )


@router.get("/history", response_model=AiAnalysisHistoryResponse)
def list_ai_analysis_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> AiAnalysisHistoryResponse:
    try:
        total, analyses = ai_analysis_service.list_history(limit=limit, offset=offset)
    except AiAnalysisPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI analysis history is unavailable.",
        ) from exc

    return AiAnalysisHistoryResponse(total=total, limit=limit, offset=offset, analyses=analyses)


@router.delete("/history/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_ai_analysis_history_item(analysis_id: int) -> None:
    try:
        deleted = ai_analysis_service.delete_history_item(analysis_id=analysis_id)
    except AiAnalysisPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI analysis history is unavailable.",
        ) from exc

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="AI analysis history item was not found.",
        )

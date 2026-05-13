from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.core.config import settings
from app.llm.langchain_client import AiAnalysisProviderError, LangChainLocalLlmClient
from app.repositories.ai_analysis_repository import AiAnalysisPersistenceError, PostgresAiAnalysisRepository
from app.schemas.ai_analysis import AiAnalysisHistoryResponse, AiAnalysisQueryRequest, AiAnalysisResponse
from app.services.ai_analysis_service import AiAnalysisService

router = APIRouter(prefix="/v1/ai-analysis", tags=["ai-analysis"])

ai_analysis_service = AiAnalysisService(
    repository=PostgresAiAnalysisRepository(settings.database_url),
    llm_client=LangChainLocalLlmClient(
        base_url=settings.local_llm_base_url,
        model=settings.local_llm_model,
        api_key=settings.local_llm_api_key,
    ),
    model=settings.local_llm_model,
    max_transactions=settings.ai_analysis_max_transactions,
)


@router.post("/query", response_model=AiAnalysisResponse)
def query_ai_analysis(payload: AiAnalysisQueryRequest) -> AiAnalysisResponse:
    try:
        return ai_analysis_service.analyze(question=payload.question, filters=payload.filters)
    except AiAnalysisProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local AI analysis is unavailable.",
        ) from exc
    except AiAnalysisPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI analysis data is unavailable.",
        ) from exc


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

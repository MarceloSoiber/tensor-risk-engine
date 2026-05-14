from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from app.llm.langchain_client import AiAnalysisLlmClient, AiAnalysisProviderError
from app.repositories.ai_analysis_repository import AiAnalysisPersistenceError, AiAnalysisRepository
from app.schemas.ai_analysis import (
    AiAnalysisDataSummary,
    AiAnalysisFilters,
    AiAnalysisHistoryItem,
    AiAnalysisResponse,
    AiAnalysisTransaction,
)


class AiAnalysisService:
    def __init__(
        self,
        *,
        repository: AiAnalysisRepository,
        llm_client: AiAnalysisLlmClient,
        model: str,
        max_transactions: int,
    ) -> None:
        self._repository = repository
        self._llm_client = llm_client
        self._model = model
        self._max_transactions = max(1, min(200, max_transactions))

    def analyze(self, *, question: str, filters: AiAnalysisFilters) -> AiAnalysisResponse:
        normalized_filters = filters.model_copy(update={"limit": min(filters.limit, self._max_transactions)})
        total, rows = self._repository.list_transactions_for_analysis(normalized_filters)
        transactions = [_normalize_transaction(row) for row in rows]
        data_summary = _build_data_summary(total=total, filters=normalized_filters, transactions=transactions)

        if not transactions:
            parsed = {
                "answer": "No transactions matched the selected filters, so no AI-supported pattern analysis was run.",
                "insights": ["No transaction context was available for the requested filters."],
                "recommended_actions": ["Adjust the filters or import/analyze transactions before running AI analysis."],
            }
        else:
            prompt = _build_prompt(question=question, data_summary=data_summary, transactions=transactions)
            try:
                raw_answer = self._llm_client.generate(prompt)
            except AiAnalysisProviderError as exc:
                self._repository.save_analysis(
                    question=question,
                    filters=normalized_filters.model_dump(),
                    transaction_ids=[transaction.id for transaction in transactions],
                    answer="Local AI analysis is unavailable.",
                    insights=[],
                    recommended_actions=[],
                    data_summary=data_summary.model_dump(mode="json"),
                    model=self._model,
                    status="failed",
                    error_message=str(exc),
                )
                raise
            parsed = _parse_llm_response(raw_answer)

        saved = self._repository.save_analysis(
            question=question,
            filters=normalized_filters.model_dump(),
            transaction_ids=[transaction.id for transaction in transactions],
            answer=parsed["answer"],
            insights=parsed["insights"],
            recommended_actions=parsed["recommended_actions"],
            data_summary=data_summary.model_dump(mode="json"),
            model=self._model,
            status="completed",
        )

        return AiAnalysisResponse(
            analysis_id=int(saved.get("id") or 0),
            question=question,
            answer=parsed["answer"],
            insights=parsed["insights"],
            recommended_actions=parsed["recommended_actions"],
            data_summary=data_summary,
            transactions=transactions,
            model=self._model,
            created_at=_coerce_datetime(saved.get("created_at")),
        )

    def list_history(self, *, limit: int, offset: int) -> tuple[int, list[AiAnalysisHistoryItem]]:
        normalized_limit = max(1, min(100, limit))
        normalized_offset = max(0, offset)
        total, rows = self._repository.list_analysis_history(limit=normalized_limit, offset=normalized_offset)
        return total, [_history_item_from_row(row) for row in rows]

    def delete_history_item(self, *, analysis_id: int) -> bool:
        if analysis_id < 1:
            return False
        return self._repository.delete_analysis_history_item(analysis_id=analysis_id)


def _build_prompt(
    *,
    question: str,
    data_summary: AiAnalysisDataSummary,
    transactions: list[AiAnalysisTransaction],
) -> str:
    transaction_context = [
        transaction.model_dump(mode="json")
        for transaction in transactions
    ]
    return f"""
You are an AI assistant supporting fraud analysts. Use only the transaction context below.
Do not invent customers, transactions, evidence, model behavior, or external facts.
Treat this as analyst assistance, not an autonomous fraud decision.
If the data is insufficient, say so clearly.
Return strict JSON with exactly these keys: answer, insights, recommended_actions.
The answer must be concise. insights and recommended_actions must be arrays of short strings.

Question:
{question}

Data summary:
{json.dumps(data_summary.model_dump(mode="json"), ensure_ascii=False)}

Transactions:
{json.dumps(transaction_context, ensure_ascii=False)}
""".strip()


def _parse_llm_response(raw_answer: str) -> dict[str, Any]:
    cleaned = _extract_json_candidate(raw_answer.strip())
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "answer": cleaned or "The local LLM returned an empty analysis.",
            "insights": [],
            "recommended_actions": ["Review the transaction context manually before taking action."],
        }

    answer = str(payload.get("answer") or "").strip() or "No answer was returned by the local LLM."
    insights = _normalize_string_list(payload.get("insights"))
    recommended_actions = _normalize_string_list(payload.get("recommended_actions"))
    return {
        "answer": answer,
        "insights": insights,
        "recommended_actions": recommended_actions,
    }


def _extract_json_candidate(value: str) -> str:
    channel_marker = "<|message|>"
    if channel_marker in value:
        value = value.split(channel_marker, maxsplit=1)[1].strip()

    start = value.find("{")
    end = value.rfind("}")
    if start != -1 and end != -1 and end > start:
        return value[start : end + 1]
    return value


def _normalize_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:10]


def _normalize_transaction(row: dict[str, object]) -> AiAnalysisTransaction:
    return AiAnalysisTransaction(
        id=int(row.get("id") or 0),
        transaction_datetime=_coerce_datetime(row.get("transaction_datetime")),
        merchant=str(row.get("merchant") or "Unknown merchant"),
        category=str(row.get("category") or "unknown"),
        amount=float(row.get("amount") or 0.0),
        risk_score=_optional_float(row.get("risk_score")),
        decision=str(row.get("decision")) if row.get("decision") is not None else None,
        is_fraud=row.get("is_fraud") if isinstance(row.get("is_fraud"), bool) else None,
        model_version=str(row.get("model_version")) if row.get("model_version") is not None else None,
    )


def _build_data_summary(
    *,
    total: int,
    filters: AiAnalysisFilters,
    transactions: list[AiAnalysisTransaction],
) -> AiAnalysisDataSummary:
    risk_scores = [transaction.risk_score for transaction in transactions if transaction.risk_score is not None]
    total_amount = sum(transaction.amount for transaction in transactions)
    average_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else None
    return AiAnalysisDataSummary(
        transaction_count=len(transactions),
        filtered_total=total,
        source=filters.source,
        decision=filters.decision,
        category=filters.category,
        average_risk_score=average_risk_score,
        total_amount=total_amount,
    )


def _history_item_from_row(row: dict[str, object]) -> AiAnalysisHistoryItem:
    data_summary = row.get("data_summary") if isinstance(row.get("data_summary"), dict) else {}
    filters = row.get("filters") if isinstance(row.get("filters"), dict) else {}
    return AiAnalysisHistoryItem(
        analysis_id=int(row.get("id") or 0),
        question=str(row.get("question") or ""),
        answer=str(row.get("answer") or ""),
        insights=_normalize_string_list(row.get("insights")),
        recommended_actions=_normalize_string_list(row.get("recommended_actions")),
        data_summary=AiAnalysisDataSummary(
            transaction_count=int(data_summary.get("transaction_count") or 0),
            filtered_total=int(data_summary.get("filtered_total") or 0),
            source=str(data_summary.get("source") or filters.get("source") or "all"),  # type: ignore[arg-type]
            decision=data_summary.get("decision") or filters.get("decision"),  # type: ignore[arg-type]
            category=data_summary.get("category") or filters.get("category"),
            average_risk_score=_optional_float(data_summary.get("average_risk_score")),
            total_amount=float(data_summary.get("total_amount") or 0.0),
        ),
        transactions=[],
        model=str(row.get("model") or ""),
        created_at=_coerce_datetime(row.get("created_at")),
        status=str(row.get("status") or "completed"),
        error_message=str(row.get("error_message")) if row.get("error_message") else None,
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _coerce_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return datetime.now(UTC)

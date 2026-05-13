from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from app.schemas.ai_analysis import AiAnalysisFilters
from app.services.dashboard_service import TransactionListFilters, TransactionSource


class AiAnalysisPersistenceError(RuntimeError):
    pass


class AiAnalysisRepository(Protocol):
    def list_transactions_for_analysis(self, filters: AiAnalysisFilters) -> tuple[int, list[dict[str, object]]]:
        ...

    def save_analysis(
        self,
        *,
        question: str,
        filters: dict[str, object],
        transaction_ids: list[int],
        answer: str,
        insights: list[str],
        recommended_actions: list[str],
        data_summary: dict[str, object],
        model: str,
        status: str,
        error_message: str | None = None,
    ) -> dict[str, object]:
        ...

    def list_analysis_history(self, *, limit: int, offset: int) -> tuple[int, list[dict[str, object]]]:
        ...


class PostgresAiAnalysisRepository:
    def __init__(self, database_url: str) -> None:
        if not database_url.strip():
            raise ValueError("database_url must not be empty.")
        self._database_url = database_url

    def list_transactions_for_analysis(self, filters: AiAnalysisFilters) -> tuple[int, list[dict[str, object]]]:
        try:
            from app.repositories.transaction_repository import TransactionPersistenceError
            from app.repositories.transaction_repository import PostgresTransactionRepository
        except ImportError as exc:
            raise AiAnalysisPersistenceError("Transaction repository is unavailable.") from exc

        repository = PostgresTransactionRepository(self._database_url)
        source = TransactionSource(filters.source)
        try:
            return repository.list_transactions(
                TransactionListFilters(
                    limit=filters.limit,
                    offset=0,
                    source=source,
                    decision=filters.decision,
                    category=filters.category,
                )
            )
        except TransactionPersistenceError as exc:
            raise AiAnalysisPersistenceError("Failed to read transaction context for AI analysis.") from exc

    def save_analysis(
        self,
        *,
        question: str,
        filters: dict[str, object],
        transaction_ids: list[int],
        answer: str,
        insights: list[str],
        recommended_actions: list[str],
        data_summary: dict[str, object],
        model: str,
        status: str,
        error_message: str | None = None,
    ) -> dict[str, object]:
        try:
            import psycopg
            from psycopg.rows import dict_row
            from psycopg.types.json import Jsonb
        except ImportError as exc:
            raise AiAnalysisPersistenceError("Postgres driver is not installed.") from exc

        try:
            with psycopg.connect(self._database_url, row_factory=dict_row) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO ai_analysis_history (
                            question,
                            filters,
                            transaction_ids,
                            answer,
                            insights,
                            recommended_actions,
                            data_summary,
                            model,
                            status,
                            error_message
                        )
                        VALUES (
                            %(question)s,
                            %(filters)s::jsonb,
                            %(transaction_ids)s,
                            %(answer)s,
                            %(insights)s::jsonb,
                            %(recommended_actions)s::jsonb,
                            %(data_summary)s::jsonb,
                            %(model)s,
                            %(status)s,
                            %(error_message)s
                        )
                        RETURNING id, created_at;
                        """,
                        {
                            "question": question,
                            "filters": Jsonb(filters),
                            "transaction_ids": transaction_ids,
                            "answer": answer,
                            "insights": Jsonb(insights),
                            "recommended_actions": Jsonb(recommended_actions),
                            "data_summary": Jsonb(data_summary),
                            "model": model,
                            "status": status,
                            "error_message": error_message,
                        },
                    )
                    return _first_row(cursor.fetchone())
        except psycopg.Error as exc:
            raise AiAnalysisPersistenceError("Failed to save AI analysis.") from exc

    def list_analysis_history(self, *, limit: int, offset: int) -> tuple[int, list[dict[str, object]]]:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise AiAnalysisPersistenceError("Postgres driver is not installed.") from exc

        try:
            with psycopg.connect(self._database_url, row_factory=dict_row) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*)::int AS total FROM ai_analysis_history;")
                    total = int(_first_row(cursor.fetchone()).get("total") or 0)
                    cursor.execute(
                        """
                        SELECT
                            id,
                            question,
                            filters,
                            transaction_ids,
                            answer,
                            insights,
                            recommended_actions,
                            data_summary,
                            model,
                            status,
                            error_message,
                            created_at
                        FROM ai_analysis_history
                        ORDER BY created_at DESC, id DESC
                        LIMIT %(limit)s OFFSET %(offset)s;
                        """,
                        {"limit": limit, "offset": offset},
                    )
                    rows = list(cursor.fetchall())
        except psycopg.Error as exc:
            raise AiAnalysisPersistenceError("Failed to list AI analysis history.") from exc

        return total, rows


def _first_row(row: dict[str, Any] | None) -> dict[str, Any]:
    return row if row is not None else {}

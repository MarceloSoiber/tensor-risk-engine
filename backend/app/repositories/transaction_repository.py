from datetime import datetime
from typing import Any, Protocol

from app.domain.decision import Decision
from app.domain.risk_score import RiskScore
from app.domain.transaction import Transaction
from app.services.dashboard_service import TransactionListFilters, TransactionSource


class TransactionPersistenceError(RuntimeError):
    pass


class TransactionRepository(Protocol):
    def save_analysis(
        self,
        transaction: Transaction,
        risk_score: RiskScore,
        decision: Decision,
        model_version: str,
    ) -> None:
        ...

    def fetch_dashboard_overview_rows(self, *, start_at: datetime, limit: int) -> dict[str, object]:
        ...

    def list_transactions(self, filters: TransactionListFilters) -> tuple[int, list[dict[str, object]]]:
        ...


class InMemoryTransactionRepository:
    def __init__(self) -> None:
        self._analyses: list[dict[str, object]] = []

    def save_analysis(
        self,
        transaction: Transaction,
        risk_score: RiskScore,
        decision: Decision,
        model_version: str,
    ) -> None:
        self._analyses.append(
            {
                "transaction": transaction,
                "risk_score": risk_score,
                "decision": decision,
                "model_version": model_version,
            }
        )

    def fetch_dashboard_overview_rows(self, *, start_at: datetime, limit: int) -> dict[str, object]:  # noqa: ARG002
        return {
            "kpis": {},
            "decision_breakdown": [],
            "fraud_trend": [],
            "category_risk": [],
            "hourly_activity": [],
            "recent_transactions": [],
        }

    def list_transactions(self, filters: TransactionListFilters) -> tuple[int, list[dict[str, object]]]:  # noqa: ARG002
        return 0, []


class PostgresTransactionRepository:
    def __init__(self, database_url: str) -> None:
        if not database_url.strip():
            raise ValueError("database_url must not be empty.")
        self._database_url = database_url

    def save_analysis(
        self,
        transaction: Transaction,
        risk_score: RiskScore,
        decision: Decision,
        model_version: str,
    ) -> None:
        try:
            import psycopg
        except ImportError as exc:
            raise TransactionPersistenceError("Postgres driver is not installed.") from exc

        try:
            with psycopg.connect(self._database_url) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO transactions (
                            transaction_datetime,
                            merchant,
                            category,
                            amount,
                            gender,
                            state,
                            city_population,
                            job,
                            customer_latitude,
                            customer_longitude,
                            merchant_latitude,
                            merchant_longitude,
                            transactions_last_hour,
                            transactions_last_24h,
                            average_amount_24h,
                            risk_score,
                            decision,
                            reasons,
                            model_version
                        )
                        VALUES (
                            %(transaction_datetime)s,
                            %(merchant)s,
                            %(category)s,
                            %(amount)s,
                            %(gender)s,
                            %(state)s,
                            %(city_population)s,
                            %(job)s,
                            %(customer_latitude)s,
                            %(customer_longitude)s,
                            %(merchant_latitude)s,
                            %(merchant_longitude)s,
                            %(transactions_last_hour)s,
                            %(transactions_last_24h)s,
                            %(average_amount_24h)s,
                            %(risk_score)s,
                            %(decision)s,
                            %(reasons)s::text[],
                            %(model_version)s
                        );
                        """,
                        {
                            "transaction_datetime": transaction.transaction_datetime,
                            "merchant": transaction.merchant,
                            "category": transaction.category,
                            "amount": transaction.amount,
                            "gender": transaction.gender,
                            "state": transaction.state,
                            "city_population": transaction.city_population,
                            "job": transaction.job,
                            "customer_latitude": transaction.customer_latitude,
                            "customer_longitude": transaction.customer_longitude,
                            "merchant_latitude": transaction.merchant_latitude,
                            "merchant_longitude": transaction.merchant_longitude,
                            "transactions_last_hour": transaction.transactions_last_hour,
                            "transactions_last_24h": transaction.transactions_last_24h,
                            "average_amount_24h": transaction.average_amount_24h,
                            "risk_score": risk_score.value,
                            "decision": decision.outcome.value,
                            "reasons": decision.reasons,
                            "model_version": model_version,
                        },
                    )
        except psycopg.Error as exc:
            raise TransactionPersistenceError("Failed to persist transaction analysis.") from exc

    def fetch_dashboard_overview_rows(self, *, start_at: datetime, limit: int) -> dict[str, object]:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise TransactionPersistenceError("Postgres driver is not installed.") from exc

        try:
            with psycopg.connect(self._database_url, row_factory=dict_row) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT
                            COUNT(*)::int AS total_transactions,
                            COUNT(*) FILTER (
                                WHERE risk_score IS NOT NULL OR decision IS NOT NULL
                            )::int AS analyzed_transactions,
                            COUNT(*) FILTER (WHERE is_fraud IS NOT NULL)::int AS known_labeled_transactions,
                            COUNT(*) FILTER (WHERE is_fraud IS TRUE)::int AS known_fraud_transactions,
                            COUNT(*) FILTER (WHERE decision IN ('review', 'reject'))::int AS reject_review_transactions,
                            COUNT(*) FILTER (WHERE risk_score IS NOT NULL)::int AS scored_transactions,
                            COALESCE(SUM(risk_score) FILTER (WHERE risk_score IS NOT NULL), 0)::float AS risk_score_sum
                        FROM transactions
                        WHERE transaction_datetime >= %(start_at)s;
                        """,
                        {"start_at": start_at},
                    )
                    kpis = _first_row(cursor.fetchone())

                    cursor.execute(
                        """
                        SELECT decision, COUNT(*)::int AS count
                        FROM transactions
                        WHERE transaction_datetime >= %(start_at)s
                          AND decision IS NOT NULL
                        GROUP BY decision
                        ORDER BY decision;
                        """,
                        {"start_at": start_at},
                    )
                    decision_breakdown = list(cursor.fetchall())

                    cursor.execute(
                        """
                        SELECT
                            transaction_datetime::date::text AS date,
                            COUNT(*)::int AS total,
                            COUNT(*) FILTER (WHERE is_fraud IS TRUE)::int AS known_fraud_count,
                            COUNT(*) FILTER (WHERE risk_score IS NOT NULL OR decision IS NOT NULL)::int AS analyzed_count,
                            COUNT(*) FILTER (WHERE decision = 'reject')::int AS rejected_count
                        FROM transactions
                        WHERE transaction_datetime >= %(start_at)s
                        GROUP BY transaction_datetime::date
                        ORDER BY transaction_datetime::date;
                        """,
                        {"start_at": start_at},
                    )
                    fraud_trend = list(cursor.fetchall())

                    cursor.execute(
                        """
                        SELECT
                            category,
                            COUNT(*)::int AS total,
                            COUNT(*) FILTER (WHERE is_fraud IS TRUE)::int AS fraud_count,
                            COUNT(*) FILTER (WHERE risk_score IS NOT NULL)::int AS scored,
                            COALESCE(SUM(risk_score) FILTER (WHERE risk_score IS NOT NULL), 0)::float AS risk_score_sum
                        FROM transactions
                        WHERE transaction_datetime >= %(start_at)s
                        GROUP BY category
                        ORDER BY total DESC, category
                        LIMIT 8;
                        """,
                        {"start_at": start_at},
                    )
                    category_risk = list(cursor.fetchall())

                    cursor.execute(
                        """
                        SELECT EXTRACT(HOUR FROM transaction_datetime)::int AS hour, COUNT(*)::int AS total
                        FROM transactions
                        WHERE transaction_datetime >= %(start_at)s
                        GROUP BY hour
                        ORDER BY hour;
                        """,
                        {"start_at": start_at},
                    )
                    hourly_activity = list(cursor.fetchall())

                    cursor.execute(
                        """
                        SELECT
                            id,
                            transaction_datetime,
                            merchant,
                            category,
                            amount::float AS amount,
                            risk_score::float AS risk_score,
                            decision,
                            is_fraud,
                            model_version
                        FROM transactions
                        WHERE transaction_datetime >= %(start_at)s
                        ORDER BY transaction_datetime DESC, id DESC
                        LIMIT %(limit)s;
                        """,
                        {"start_at": start_at, "limit": limit},
                    )
                    recent_transactions = list(cursor.fetchall())
        except psycopg.Error as exc:
            raise TransactionPersistenceError("Failed to read dashboard data.") from exc

        return {
            "kpis": kpis,
            "decision_breakdown": decision_breakdown,
            "fraud_trend": fraud_trend,
            "category_risk": category_risk,
            "hourly_activity": hourly_activity,
            "recent_transactions": recent_transactions,
        }

    def list_transactions(self, filters: TransactionListFilters) -> tuple[int, list[dict[str, object]]]:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise TransactionPersistenceError("Postgres driver is not installed.") from exc

        where_sql, params = _build_transaction_filters(filters)

        try:
            with psycopg.connect(self._database_url, row_factory=dict_row) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(f"SELECT COUNT(*)::int AS total FROM transactions {where_sql};", params)
                    total = int(_first_row(cursor.fetchone()).get("total") or 0)
                    cursor.execute(
                        f"""
                        SELECT
                            id,
                            transaction_datetime,
                            merchant,
                            category,
                            amount::float AS amount,
                            risk_score::float AS risk_score,
                            decision,
                            is_fraud,
                            model_version
                        FROM transactions
                        {where_sql}
                        ORDER BY transaction_datetime DESC, id DESC
                        LIMIT %(limit)s OFFSET %(offset)s;
                        """,
                        {**params, "limit": filters.limit, "offset": filters.offset},
                    )
                    rows = list(cursor.fetchall())
        except psycopg.Error as exc:
            raise TransactionPersistenceError("Failed to list transactions.") from exc

        return total, rows


def _first_row(row: dict[str, Any] | None) -> dict[str, Any]:
    return row if row is not None else {}


def _build_transaction_filters(filters: TransactionListFilters) -> tuple[str, dict[str, object]]:
    clauses: list[str] = []
    params: dict[str, object] = {}

    if filters.source == TransactionSource.ANALYZED:
        clauses.append("(risk_score IS NOT NULL OR decision IS NOT NULL)")
    elif filters.source == TransactionSource.HISTORICAL:
        clauses.append("is_fraud IS NOT NULL")

    if filters.decision is not None:
        clauses.append("decision = %(decision)s")
        params["decision"] = filters.decision

    if filters.category is not None:
        clauses.append("category = %(category)s")
        params["category"] = filters.category

    if not clauses:
        return "", params
    return f"WHERE {' AND '.join(clauses)}", params

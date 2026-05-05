from typing import Protocol

from app.domain.decision import Decision
from app.domain.risk_score import RiskScore
from app.domain.transaction import Transaction


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

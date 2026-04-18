from typing import Protocol

from app.domain.transaction import Transaction


class TransactionRepository(Protocol):
    def save(self, transaction: Transaction) -> None:
        ...


class InMemoryTransactionRepository:
    def __init__(self) -> None:
        self._transactions: list[Transaction] = []

    def save(self, transaction: Transaction) -> None:
        self._transactions.append(transaction)

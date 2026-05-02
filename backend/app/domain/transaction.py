from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Transaction:
    amount: float
    transaction_datetime: datetime
    merchant: str
    category: str
    gender: str
    state: str
    job: str
    city_population: int
    customer_latitude: float
    customer_longitude: float
    merchant_latitude: float
    merchant_longitude: float
    transactions_last_hour: int
    transactions_last_24h: int
    average_amount_24h: float

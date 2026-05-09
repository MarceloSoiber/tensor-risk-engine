from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_KAGGLE_COLUMNS = {
    "Unnamed: 0",
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud",
}


@dataclass(frozen=True)
class ImportResult:
    processed_rows: int
    imported_rows: int


class TransactionImportError(RuntimeError):
    pass


class TransactionImportService:
    def __init__(self, database_url: str) -> None:
        if not database_url.strip():
            raise ValueError("database_url must not be empty.")
        self._database_url = database_url

    def import_csv(self, *, dataset_path: Path, batch_size: int) -> ImportResult:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        try:
            import psycopg
        except ImportError as exc:
            raise TransactionImportError("Postgres driver is not installed.") from exc

        processed_rows = 0
        imported_rows = 0

        try:
            for frame in pd.read_csv(dataset_path, chunksize=batch_size):
                self._validate_columns(frame.columns)
                rows = [_map_kaggle_row(row) for row in frame.to_dict(orient="records")]
                processed_rows += len(rows)
                if not rows:
                    continue
                with psycopg.connect(self._database_url) as connection:
                    with connection.cursor() as cursor:
                        cursor.executemany(_UPSERT_SQL, rows)
                    imported_rows += len(rows)
        except pd.errors.ParserError as exc:
            raise TransactionImportError("Dataset could not be parsed as CSV.") from exc
        except psycopg.Error as exc:
            raise TransactionImportError("Failed to import transactions.") from exc

        return ImportResult(processed_rows=processed_rows, imported_rows=imported_rows)

    @staticmethod
    def _validate_columns(columns: Iterable[str]) -> None:
        missing = sorted(REQUIRED_KAGGLE_COLUMNS.difference(set(columns)))
        if missing:
            raise TransactionImportError(f"Dataset is missing required columns: {', '.join(missing)}")


def _map_kaggle_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "source_row_number": _optional_int(row["Unnamed: 0"]),
        "transaction_datetime": datetime.strptime(str(row["trans_date_trans_time"]), "%Y-%m-%d %H:%M:%S"),
        "card_number": str(row["cc_num"]),
        "merchant": str(row["merchant"]),
        "category": str(row["category"]),
        "amount": float(row["amt"]),
        "first_name": _optional_text(row["first"]),
        "last_name": _optional_text(row["last"]),
        "gender": _optional_text(row["gender"]),
        "street": _optional_text(row["street"]),
        "city": _optional_text(row["city"]),
        "state": _optional_text(row["state"]),
        "postal_code": _optional_text(row["zip"]),
        "customer_latitude": float(row["lat"]),
        "customer_longitude": float(row["long"]),
        "city_population": _optional_int(row["city_pop"]),
        "job": _optional_text(row["job"]),
        "date_of_birth": datetime.strptime(str(row["dob"]), "%Y-%m-%d").date(),
        "transaction_number": str(row["trans_num"]),
        "unix_time": _optional_int(row["unix_time"]),
        "merchant_latitude": float(row["merch_lat"]),
        "merchant_longitude": float(row["merch_long"]),
        "is_fraud": bool(int(row["is_fraud"])),
    }


def _optional_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


_UPSERT_SQL = """
INSERT INTO transactions (
    source_row_number,
    transaction_datetime,
    card_number,
    merchant,
    category,
    amount,
    first_name,
    last_name,
    gender,
    street,
    city,
    state,
    postal_code,
    customer_latitude,
    customer_longitude,
    city_population,
    job,
    date_of_birth,
    transaction_number,
    unix_time,
    merchant_latitude,
    merchant_longitude,
    is_fraud
)
VALUES (
    %(source_row_number)s,
    %(transaction_datetime)s,
    %(card_number)s,
    %(merchant)s,
    %(category)s,
    %(amount)s,
    %(first_name)s,
    %(last_name)s,
    %(gender)s,
    %(street)s,
    %(city)s,
    %(state)s,
    %(postal_code)s,
    %(customer_latitude)s,
    %(customer_longitude)s,
    %(city_population)s,
    %(job)s,
    %(date_of_birth)s,
    %(transaction_number)s,
    %(unix_time)s,
    %(merchant_latitude)s,
    %(merchant_longitude)s,
    %(is_fraud)s
)
ON CONFLICT (transaction_number) WHERE transaction_number IS NOT NULL DO UPDATE SET
    source_row_number = EXCLUDED.source_row_number,
    transaction_datetime = EXCLUDED.transaction_datetime,
    card_number = EXCLUDED.card_number,
    merchant = EXCLUDED.merchant,
    category = EXCLUDED.category,
    amount = EXCLUDED.amount,
    first_name = EXCLUDED.first_name,
    last_name = EXCLUDED.last_name,
    gender = EXCLUDED.gender,
    street = EXCLUDED.street,
    city = EXCLUDED.city,
    state = EXCLUDED.state,
    postal_code = EXCLUDED.postal_code,
    customer_latitude = EXCLUDED.customer_latitude,
    customer_longitude = EXCLUDED.customer_longitude,
    city_population = EXCLUDED.city_population,
    job = EXCLUDED.job,
    date_of_birth = EXCLUDED.date_of_birth,
    unix_time = EXCLUDED.unix_time,
    merchant_latitude = EXCLUDED.merchant_latitude,
    merchant_longitude = EXCLUDED.merchant_longitude,
    is_fraud = EXCLUDED.is_fraud,
    updated_at = NOW();
"""

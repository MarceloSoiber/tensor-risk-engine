from __future__ import annotations

from app.services.transaction_import_service import (
    REQUIRED_KAGGLE_COLUMNS,
    TransactionImportError,
    TransactionImportService,
    _map_kaggle_row,
    _transaction_from_import_row,
)


def test_import_service_validates_required_columns() -> None:
    columns = REQUIRED_KAGGLE_COLUMNS - {"trans_num"}

    try:
        TransactionImportService._validate_columns(columns)
    except TransactionImportError as exc:
        assert "trans_num" in str(exc)
    else:
        raise AssertionError("Expected missing column validation error.")


def test_import_mapper_builds_transaction_for_analysis() -> None:
    mapped = _map_kaggle_row(
        {
            "Unnamed: 0": 1,
            "trans_date_trans_time": "2020-06-21 12:14:25",
            "cc_num": 4000000000000000,
            "merchant": "fraud_Test Merchant",
            "category": "grocery_pos",
            "amt": 43.20,
            "first": "Jane",
            "last": "Doe",
            "gender": "F",
            "street": "1 Main St",
            "city": "Austin",
            "state": "TX",
            "zip": "78701",
            "lat": 30.2672,
            "long": -97.7431,
            "city_pop": 950000,
            "job": "Engineer",
            "dob": "1990-01-01",
            "trans_num": "test-tx-1",
            "unix_time": 1592741665,
            "merch_lat": 30.3000,
            "merch_long": -97.7000,
            "is_fraud": 0,
        }
    )

    transaction = _transaction_from_import_row(mapped)

    assert transaction.amount == 43.20
    assert transaction.merchant == "fraud_Test Merchant"
    assert transaction.category == "grocery_pos"
    assert transaction.transactions_last_hour == 0
    assert transaction.transactions_last_24h == 1
    assert transaction.average_amount_24h == 43.20

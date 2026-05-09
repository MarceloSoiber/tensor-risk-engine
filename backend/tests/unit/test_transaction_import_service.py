from __future__ import annotations

from app.services.transaction_import_service import REQUIRED_KAGGLE_COLUMNS, TransactionImportError, TransactionImportService


def test_import_service_validates_required_columns() -> None:
    columns = REQUIRED_KAGGLE_COLUMNS - {"trans_num"}

    try:
        TransactionImportService._validate_columns(columns)
    except TransactionImportError as exc:
        assert "trans_num" in str(exc)
    else:
        raise AssertionError("Expected missing column validation error.")

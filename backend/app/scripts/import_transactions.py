from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import settings
from app.services.transaction_import_service import TransactionImportService


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Kaggle fraud transactions into PostgreSQL.")
    parser.add_argument("--dataset", required=True, help="Path to fraudTrain.csv or fraudTest.csv.")
    parser.add_argument("--batch-size", type=int, default=5000, help="Number of CSV rows imported per batch.")
    args = parser.parse_args()

    service = TransactionImportService(settings.database_url)
    result = service.import_csv(dataset_path=Path(args.dataset), batch_size=args.batch_size)
    print(f"Processed {result.processed_rows} rows. Imported {result.imported_rows} rows.")


if __name__ == "__main__":
    main()

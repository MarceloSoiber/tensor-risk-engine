BEGIN;

ALTER TABLE transactions
    ADD COLUMN IF NOT EXISTS transactions_last_hour INTEGER,
    ADD COLUMN IF NOT EXISTS transactions_last_24h INTEGER,
    ADD COLUMN IF NOT EXISTS average_amount_24h NUMERIC(12, 2);

ALTER TABLE transactions
    ADD CONSTRAINT transactions_last_hour_non_negative
        CHECK (transactions_last_hour IS NULL OR transactions_last_hour >= 0),
    ADD CONSTRAINT transactions_last_24h_non_negative
        CHECK (transactions_last_24h IS NULL OR transactions_last_24h >= 0),
    ADD CONSTRAINT transactions_last_hour_not_greater_than_24h
        CHECK (
            transactions_last_hour IS NULL
            OR transactions_last_24h IS NULL
            OR transactions_last_hour <= transactions_last_24h
        ),
    ADD CONSTRAINT transactions_average_amount_24h_non_negative
        CHECK (average_amount_24h IS NULL OR average_amount_24h >= 0);

COMMIT;

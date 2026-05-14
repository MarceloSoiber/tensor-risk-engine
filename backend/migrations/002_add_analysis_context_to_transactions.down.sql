BEGIN;

ALTER TABLE transactions
    DROP CONSTRAINT IF EXISTS transactions_average_amount_24h_non_negative,
    DROP CONSTRAINT IF EXISTS transactions_last_hour_not_greater_than_24h,
    DROP CONSTRAINT IF EXISTS transactions_last_24h_non_negative,
    DROP CONSTRAINT IF EXISTS transactions_last_hour_non_negative,
    DROP COLUMN IF EXISTS average_amount_24h,
    DROP COLUMN IF EXISTS transactions_last_24h,
    DROP COLUMN IF EXISTS transactions_last_hour;

COMMIT;

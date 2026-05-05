BEGIN;

ALTER TABLE transactions
    ALTER COLUMN card_number DROP NOT NULL;

COMMIT;

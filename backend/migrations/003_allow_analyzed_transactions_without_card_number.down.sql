BEGIN;

ALTER TABLE transactions
    ALTER COLUMN card_number SET NOT NULL;

COMMIT;

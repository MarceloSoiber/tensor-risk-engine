BEGIN;

DROP INDEX IF EXISTS transactions_is_fraud_idx;
DROP INDEX IF EXISTS transactions_decision_idx;
DROP INDEX IF EXISTS transactions_card_time_idx;
DROP INDEX IF EXISTS transactions_transaction_number_uidx;
DROP TABLE IF EXISTS transactions;

COMMIT;

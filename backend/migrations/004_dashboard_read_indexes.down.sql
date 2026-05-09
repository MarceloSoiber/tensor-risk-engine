BEGIN;

DROP INDEX IF EXISTS transactions_risk_score_time_idx;
DROP INDEX IF EXISTS transactions_is_fraud_time_idx;
DROP INDEX IF EXISTS transactions_decision_time_idx;
DROP INDEX IF EXISTS transactions_category_idx;
DROP INDEX IF EXISTS transactions_created_at_idx;
DROP INDEX IF EXISTS transactions_transaction_datetime_idx;

COMMIT;

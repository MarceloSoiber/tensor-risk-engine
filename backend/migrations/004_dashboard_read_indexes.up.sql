BEGIN;

CREATE INDEX IF NOT EXISTS transactions_transaction_datetime_idx
    ON transactions (transaction_datetime);

CREATE INDEX IF NOT EXISTS transactions_created_at_idx
    ON transactions (created_at);

CREATE INDEX IF NOT EXISTS transactions_category_idx
    ON transactions (category);

CREATE INDEX IF NOT EXISTS transactions_decision_time_idx
    ON transactions (decision, transaction_datetime);

CREATE INDEX IF NOT EXISTS transactions_is_fraud_time_idx
    ON transactions (is_fraud, transaction_datetime);

CREATE INDEX IF NOT EXISTS transactions_risk_score_time_idx
    ON transactions (risk_score, transaction_datetime);

COMMIT;

BEGIN;

CREATE TABLE IF NOT EXISTS transactions (
    id BIGSERIAL PRIMARY KEY,
    source_row_number BIGINT,
    transaction_datetime TIMESTAMP NOT NULL,
    card_number VARCHAR(32) NOT NULL,
    merchant VARCHAR(128) NOT NULL,
    category VARCHAR(100) NOT NULL,
    amount NUMERIC(12, 2) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    gender CHAR(1),
    street VARCHAR(255),
    city VARCHAR(128),
    state CHAR(2),
    postal_code VARCHAR(16),
    customer_latitude DOUBLE PRECISION NOT NULL,
    customer_longitude DOUBLE PRECISION NOT NULL,
    city_population INTEGER,
    job VARCHAR(128),
    date_of_birth DATE,
    transaction_number VARCHAR(64),
    unix_time BIGINT,
    merchant_latitude DOUBLE PRECISION NOT NULL,
    merchant_longitude DOUBLE PRECISION NOT NULL,
    is_fraud BOOLEAN,
    risk_score NUMERIC(18, 17),
    decision VARCHAR(16),
    reasons TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    model_version VARCHAR(160),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT transactions_amount_non_negative CHECK (amount >= 0),
    CONSTRAINT transactions_gender_valid CHECK (gender IS NULL OR gender IN ('F', 'M')),
    CONSTRAINT transactions_coordinates_valid CHECK (
        customer_latitude BETWEEN -90 AND 90
        AND merchant_latitude BETWEEN -90 AND 90
        AND customer_longitude BETWEEN -180 AND 180
        AND merchant_longitude BETWEEN -180 AND 180
    ),
    CONSTRAINT transactions_city_population_non_negative CHECK (
        city_population IS NULL OR city_population >= 0
    ),
    CONSTRAINT transactions_risk_score_valid CHECK (
        risk_score IS NULL OR risk_score BETWEEN 0 AND 1
    ),
    CONSTRAINT transactions_decision_valid CHECK (
        decision IS NULL OR decision IN ('approve', 'review', 'reject')
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS transactions_transaction_number_uidx
    ON transactions (transaction_number)
    WHERE transaction_number IS NOT NULL;

CREATE INDEX IF NOT EXISTS transactions_card_time_idx
    ON transactions (card_number, transaction_datetime);

CREATE INDEX IF NOT EXISTS transactions_decision_idx
    ON transactions (decision);

CREATE INDEX IF NOT EXISTS transactions_is_fraud_idx
    ON transactions (is_fraud);

COMMIT;

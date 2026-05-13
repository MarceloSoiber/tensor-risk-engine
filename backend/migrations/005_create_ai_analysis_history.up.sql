BEGIN;

CREATE TABLE IF NOT EXISTS ai_analysis_history (
    id BIGSERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    filters JSONB NOT NULL DEFAULT '{}'::JSONB,
    transaction_ids BIGINT[] NOT NULL DEFAULT ARRAY[]::BIGINT[],
    answer TEXT NOT NULL,
    insights JSONB NOT NULL DEFAULT '[]'::JSONB,
    recommended_actions JSONB NOT NULL DEFAULT '[]'::JSONB,
    data_summary JSONB NOT NULL DEFAULT '{}'::JSONB,
    model VARCHAR(160) NOT NULL,
    status VARCHAR(24) NOT NULL,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT ai_analysis_history_status_valid CHECK (status IN ('completed', 'failed'))
);

CREATE INDEX IF NOT EXISTS ai_analysis_history_created_at_idx
    ON ai_analysis_history (created_at DESC);

COMMIT;

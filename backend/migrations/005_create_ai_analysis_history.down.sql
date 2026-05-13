BEGIN;

DROP INDEX IF EXISTS ai_analysis_history_created_at_idx;
DROP TABLE IF EXISTS ai_analysis_history;

COMMIT;

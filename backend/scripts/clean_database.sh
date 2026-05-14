#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.database.yml}"
POSTGRES_SERVICE="${POSTGRES_SERVICE:-postgres}"

docker compose -f "$COMPOSE_FILE" exec -T "$POSTGRES_SERVICE" sh -c '
  psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
    -c "TRUNCATE TABLE transactions, ai_analysis_history RESTART IDENTITY CASCADE;"
'

echo "Database data was cleaned successfully, including imported transactions."

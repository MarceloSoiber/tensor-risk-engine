#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${BACKEND_ROOT}/.." && pwd)"
MIGRATIONS_DIR="${BACKEND_ROOT}/migrations"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.database.yml"
POSTGRES_SERVICE="${POSTGRES_SERVICE:-postgres}"
POSTGRES_DB="${POSTGRES_DB:-fraud_detection}"
POSTGRES_USER="${POSTGRES_USER:-fraud_user}"
COMMAND="${1:-up}"

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required to run database migrations." >&2
    exit 1
fi

if [ ! -f "${COMPOSE_FILE}" ]; then
    echo "Database compose file not found: ${COMPOSE_FILE}" >&2
    exit 1
fi

if [ ! -d "${MIGRATIONS_DIR}" ]; then
    echo "Migrations directory not found: ${MIGRATIONS_DIR}" >&2
    exit 1
fi

run_psql() {
    docker compose -f "${COMPOSE_FILE}" exec -T "${POSTGRES_SERVICE}" \
        psql -v ON_ERROR_STOP=1 -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" "$@"
}

ensure_migrations_table() {
    run_psql <<'SQL'
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
SQL
}

is_applied() {
    local version="$1"

    run_psql -tAc "SELECT 1 FROM schema_migrations WHERE version = '${version}' LIMIT 1;" \
        </dev/null \
        | tr -d '[:space:]'
}

record_applied() {
    local version="$1"

    run_psql -c "INSERT INTO schema_migrations (version) VALUES ('${version}');" </dev/null >/dev/null
}

remove_applied() {
    local version="$1"

    run_psql -c "DELETE FROM schema_migrations WHERE version = '${version}';" </dev/null >/dev/null
}

apply_migration() {
    local migration_file="$1"
    local migration_name
    local version

    migration_name="$(basename "${migration_file}")"
    version="${migration_name%.up.sql}"

    if [ "$(is_applied "${version}")" = "1" ]; then
        echo "Skipping already applied migration: ${version}"
        return
    fi

    echo "Applying migration: ${version}"
    run_psql <"${migration_file}"
    record_applied "${version}"
    echo "Applied migration: ${version}"
}

rollback_migration() {
    local migration_file="$1"
    local migration_name
    local version

    migration_name="$(basename "${migration_file}")"
    version="${migration_name%.down.sql}"

    if [ "$(is_applied "${version}")" != "1" ]; then
        echo "Skipping unapplied migration rollback: ${version}"
        return
    fi

    echo "Rolling back migration: ${version}"
    run_psql <"${migration_file}"
    remove_applied "${version}"
    echo "Rolled back migration: ${version}"
}

run_up() {
    local migration_count=0
    local migration_file

    ensure_migrations_table

    while IFS= read -r migration_file; do
        migration_count=$((migration_count + 1))
        apply_migration "${migration_file}"
    done < <(find "${MIGRATIONS_DIR}" -maxdepth 1 -type f -name "*.up.sql" | sort)

    if [ "${migration_count}" -eq 0 ]; then
        echo "No migration files found in ${MIGRATIONS_DIR}."
        return
    fi

    echo "Database migrations finished."
}

run_down() {
    local migration_count=0
    local migration_file

    ensure_migrations_table

    while IFS= read -r migration_file; do
        migration_count=$((migration_count + 1))
        rollback_migration "${migration_file}"
    done < <(find "${MIGRATIONS_DIR}" -maxdepth 1 -type f -name "*.down.sql" | sort -r)

    if [ "${migration_count}" -eq 0 ]; then
        echo "No rollback migration files found in ${MIGRATIONS_DIR}."
        return
    fi

    echo "Database rollbacks finished."
}

main() {
    case "${COMMAND}" in
        up)
            run_up
            ;;
        down)
            run_down
            ;;
        *)
            echo "Usage: $0 [up|down]" >&2
            exit 1
            ;;
    esac
}

main "$@"
